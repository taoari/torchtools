import torch
from .hooks import register_forward_hooks

import sys
sys.setrecursionlimit(6000) # default 3000

# def chunk_string(s, length):
#     return [s[i:length + i] for i in range(0, len(s), length)]

# def _beautify(name):
#     return '\n'.join(chunk_string(name, 20))

def _collect_tensors(out):
    if torch.is_tensor(out):
        yield out
    elif isinstance(out, (tuple, list)):
        for _o in out:
            for __o in _collect_tensors(_o):
                yield __o
    elif isinstance(out, dict):
        for _o in out.values():
            for __o in _collect_tensors(_o):
                yield __o
    else:
        import warnings
        warnings.warn('{} can not be colleted as a tensor'.format(out))

def _create_tensors(shape, device='cpu'):
    if isinstance(shape, (tuple, list)):
        if isinstance(shape[0], int):
            return torch.randn(*shape, device=device)
        else:
            return list([_create_tensors(_sh) for _sh in shape])
    elif isinstance(shape, dict):
        return {k:_create_tensors(_sh) for k, _sh in shape.items()}


def _get_module_info(mod):
    # format: kxk/s/p/d:<dilation>/g:<group>
    strs = []
    try:
        if hasattr(mod, 'kernel_size'):
            strs.append('x'.join(map(str, mod.kernel_size)))
        if hasattr(mod, 'stride'):
            strs.append(str(mod.stride[0]) if mod.stride[0] == mod.stride[1] else 'x'.join(map(str, mod.stride)))
        if hasattr(mod, 'padding'):
            strs.append(str(mod.padding[0]) if mod.padding[0] == mod.padding[1] else 'x'.join(map(str, mod.padding)))
        if hasattr(mod, 'dilation'):
            if max(mod.dilation) > 1:
                strs.append('d:'+str(mod.dilation[0]) if mod.dilation[0] == mod.dilation[1] else 'x'.join(map(str, mod.dilation)))
        if hasattr(mod, 'groups'):
            if mod.groups > 1:
                strs.append('g:'+str(mod.groups))
        if hasattr(mod, 'inplace') and mod.inplace:
            strs.append('inplace')
    except Exception as e:
        print(e)
    return '/'.join(strs)


def _get_info_grad_fn(model, inputs, output):
    # trace back to construct the computational graph

    def add_node(grad_fn):
        # print(grad_fn)
        if hasattr(grad_fn, 'variable'):
            # DELEGATE AccumulateGrad to Parameter
            param = grad_fn.variable
            info['nodes'][str(id(param))].update(dict(_type='PARAMETER',
                _class=type(param).__name__, shape=list(param.shape)))
        else:
            info['nodes'][str(id(grad_fn))].update(dict(_type='BACKWARD', _class=type(grad_fn).__name__))

    seen = set()
    def add_edges(grad_fn):
        if grad_fn is None or grad_fn in seen:
            return
        add_node(grad_fn)
        seen.add(grad_fn)
        assert hasattr(grad_fn, 'next_functions')

        for gf, _ in grad_fn.next_functions:
            if gf is not None:
                if hasattr(gf, 'variable'):
                    # DELEGATE AccumulateGrad to Parameter
                    info['edges'][str((id(gf.variable)))].append(str(id(grad_fn)))
                else:
                    info['edges'][str((id(gf)))].append(str(id(grad_fn)))
                add_edges(gf)

    from collections import defaultdict
    info = {'nodes': defaultdict(dict), 'edges': defaultdict(list)}
    # info = {'nodes': dict of dict, 'edges': dict of list (graph adjacency list)}
    # info['nodes'][i] = {...}          # node_attrs
    # info['edges'][i] = [j, k, ...]    # adjacent list

    for out in _collect_tensors(output):
        add_edges(out.grad_fn)
        info['edges'][str(id(out.grad_fn))].append(str(id(out)))
        info['nodes'][str(id(out))].update(dict(_type='OUTPUT'), _class='Output', shape=list(out.shape))
    # NOTE: inputs and model are only used here
    for in_ in _collect_tensors(inputs):
        info['nodes'][str(id(in_))].update(dict(_type='INPUT'), _class='Input', shape=list(in_.shape))
    for name, param in model.named_parameters():
        # NOTE: can introduced not traced back parameters for MaskRCNN (defined but not used??)
        info['nodes'][str(id(param))].update(dict(_type='PARAMETER', _class=type(param).__name__, name=name, shape=list(param.shape)))
    return info

def _transfer_forward_hook_info(info, hook):
    # transfer module name, hyperparameters, output shape to info
    for _info in hook:
        mod, _in, _out = _info['module'], _info['inputs'], _info['output']
        if torch.is_tensor(_out): # NOTE: only for convential modules, i.e. output is single tensor
            # DELEGATE module (name, class, hyper-parameters), _out (output shape) to _out.grad_fn
            nid = str(id(_out.grad_fn))
            if nid in info['nodes']: # NOTE: only transfer those can be matched
                info['nodes'][nid].update(name=_info['module_name'], module_class=type(mod).__name__,
                    module_info=_get_module_info(mod), shape=list(_out.shape))

def _present_graph(info):
    # info = {'nodes': dict of dict, 'edges': dict of list (graph adjacency list)}
    # info['nodes'][i] = {...}          # node_attrs
    # info['edges'][i] = [j, k, ...]    # adjacent list

    def _get_label(id_, nd):
        strs = []
        if 'name' in nd:
            strs.append(nd['name'])
        if '_class' in nd:
            if nd['_class'] not in ['Parameter']: # remove Parameter as it is obvious to see from the color
                strs.append(nd['_class'].split('Backward')[0]) # remove Backward surfix
        if 'module_class' in nd:
            _extra = None
            if nd['module_class'] == 'Conv2d': # Indicate Conv2d depthwise, group, or pointwise
                if 'g:' in nd['module_info']:
                    groups = int(nd['module_info'].split('g:')[1].split('/')[0])
                    _extra = 'Depthwise' if groups == nd['shape'][1] else 'Group'
                elif '1x1' in nd['module_info']:
                    _extra = 'Pointwise'

            if _extra is not None:
                strs.append('[{} ({})]'.format(nd['module_class'], _extra))
            else:
                strs.append('[{}]'.format(nd['module_class']))
   
        # if '_type' in nd:
        #     strs.append('({})'.format(nd['_type'])) # already indicated in color
        if 'module_info' in nd and nd['module_info']:
            strs.append(nd['module_info'])
        if 'shape' in nd:
            strs.append('(' + ','.join(map(str, nd['shape'])) + ')')
        return '\n'.join(strs)

    # draw dot graph
    import graphviz
    g = graphviz.Digraph(node_attr=STYLES['DEFAULT'])
    for id_, nd in info['nodes'].items():
        g.node(id_, label=_get_label(id_, nd), **STYLES[nd['_type']])
    for src, dsts in info['edges'].items():
        for dst in dsts:
            if '_class' in info['nodes'][src] and info['nodes'][src]['_class'] in ['Parameter', 'TBackward']:
                g.edge(src, dst)
            else:
                g.edge(src, dst, weight='5') # make non-parameter edges large weight to straighten the graph
    return g

def _backup_requires_grad(model, inputs):
    # record requires_grad status for inputs and parameters
    params_status = {}
    for t in _collect_tensors(inputs):
        params_status[t] = t.requires_grad
        t.requires_grad_(True)
    for n, t in model.named_parameters():
        params_status[t] = t.requires_grad
        t.requires_grad_(True)
    return params_status

def _recover_requires_grad(params_status):
    # recover requires_grad
    for t, req_grad in params_status.items():
        try:
            t.requires_grad_(req_grad)
        except Exception as e:
            t.data.requires_grad_(req_grad)


STYLES = dict(DEFAULT=dict(shape='box'), # default style is applied to all nodes, default fontsize: 14
    INPUT=dict(style='filled', fillcolor='slateblue'),
    OUTPUT=dict(style='filled', fillcolor='slateblue'),
    INTERMEDIATE=dict(),
    PARAMETER=dict(style='filled', fillcolor='orange', fontsize='9'),
    PARAMETER_FREEZED=dict(style='filled', fillcolor='bisque', fontsize='9'), # light orange
    MODULE=dict(style='filled'), # gray
    GRADFN=dict(style='filled'),
    BACKWARD=dict(style='filled'),)
    # Backward is the same as GradFn, Backward is traced back from out.grad_fn.next_functions,
    # GradFn is from tensor.grad_fn (in forward hooks)


def plot_network(model, *inputs, output=None):
    # NOTE: 1. str(id(grad_fn)) can be changed when trace back output.grad_fn
    #          must trace back out.grad_fn first, and then call str(id(tensor.grad_fn)) can keep. WHY??
    # 2. inputs (GradFn) -> Module -> output(s) (GradFn) (forward hook)
    #         |                          ^
    #         |--------------------------|
    #             out.grad_fn
    #   DELEGATE Module, Output Tensor to Output Tensor GradFn
    # 3. AccumulateGrad (BackwardFn) -> Parameter
    #   DELEGATE AccumulateGrad to Parameter
    if output is None:
        params_status = _backup_requires_grad(model, inputs) # also set_requires_grad to True to trace back
        with register_forward_hooks(model) as forward:
            model.eval()
            with torch.set_grad_enabled(True):
                output = model(*inputs)
            info = _get_info_grad_fn(model, inputs, output)
            _transfer_forward_hook_info(info, forward) # optional (but for more module info)
        _recover_requires_grad(params_status)
    else:
        info = _get_info_grad_fn(model, inputs, output)
    # update for freezed parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            info['nodes'][str(id(param))].update(dict(_type='PARAMETER_FREEZED',
                _class=type(param).__name__, name=name))
    g = _present_graph(info)
    return g

def plot_network_tensorboard(model, *inputs, writer=None):
    model.eval()

    if writer is not None:
        writer.add_graph(model, inputs)
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
        writer.add_graph(model, inputs)
        writer.close()


def test_plot_network():

    import torch
    import torchvision.models as models

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = models.resnet18().to(device)
    inputs = torch.randn(1, 3, 224, 224).to(device)

    plot_network(model, inputs).save('resnet18.gv')

    output = model(inputs)
    plot_network(model, inputs, output=output).save('resnet18_grad_fn.gv')

if __name__ == '__main__':
    
    test_plot_network()
