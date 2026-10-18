import importlib
from os import path as osp

from basicsr.utils import scandir

# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
# arch_filenames = [
#     osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
#     if v.endswith('_linear_attention.py')
# ]

arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    # if v.endswith('strip_swa_atd_ResFFN_tiny.py')
    # if v.endswith('strip_swa_atd_laststage_convFFN_tiny.py')
    # if v.endswith('strip_swa_atd_rech4_convFFN_tiny.py')
    # if v.endswith('strip_swa_agent_nodwc_convFFN_tiny.py')
    # if v.endswith('strip_swa_agent_noposbias_convFFN_tiny.py')
    # if v.endswith('strip_swa_agent_noposbiasDWC_convFFN_tiny.py')
    # if v.endswith('strip_swa_agent_diffnum_noposbiasDWC_convFFN_tiny.py')
    # if v.endswith('strip_swa_agent_proj_noposbiasDWC_convFFN_tiny.py')
    # if v.endswith('strip_swa_agent_proj_noposbiasDWC_parallel_convFFN_tiny.py')
    # if v.endswith('strip_swa_agent_proj_noposbiasDWC_DFE_parallel_convFFN_tiny.py')
    # if v.endswith('strip_swa_agent_proj_noposbiasDWC_HAI_parallel_convFFN_tiny.py')
    # if v.endswith('strip_swa_multiscale_convFFN_tiny.py')
    # if v.endswith('strip_swa_agent_proj_noposbiasDWC_parallel2_HAI_convFFN_tiny.py')
    # if v.endswith('strip_swa_anchor_convFFN_tiny.py')
    # if v.endswith('strip_swa_dict_convFFN_tiny.py')
    # if v.endswith('dict_mlp_only_convFFN_tiny.py')
    # if v.endswith('strip_swa_dict_mlp_convFFN_tiny.py')
    # if v.endswith('atdpaper_convFFN_tiny.py')
    # if v.endswith('strip_swa_dict_mlp_parallel_convFFN_tiny.py')
    # if v.endswith('strip_swa_dict_mlp_gateFFN_tiny.py')
    # if v.endswith('strip_swa_dict_mlp_restomerFFN_tiny.py')
    # if v.endswith('strip_swa_dict_mlp_restomerFFN_small.py')
    #####dict cr model
    # if v.endswith('strip_swa_dict_mlp_restomerFFN_tiny_stage3_inoutks7.py')
    # if v.endswith('strip_swa_dict_mlp_restomerFFN_tiny_stage3_inoutks7_reducedwcq.py')
    # if v.endswith('strip_swa_dict_mlp_restomerFFN_tiny_stage3_inoutks7_reducemixmlp.py')
    # if v.endswith('strip_swa_dict_mlp_restomerFFN_tiny_stage3_inoutks7_reducedwcq_mixmlp.py')
    # if v.endswith('strip_swa_dict_mlp_restomerFFN_tiny_stage3_inoutks3_multirectsa.py')

    # if v.endswith('strip_swa_dict_mlp_restomerFFN_tiny_stage3_inoutks3_multirectsa_nodwcq_initdict.py')
    # if v.endswith('strip_swa_atd_restomerFFN_tiny_stage3_inoutks3.py')

    ###ssim test on torch 1.8.1 ,2.1.1 donot 2.4
    if v.endswith('strip_swa_dict_restomerFFN_tiny_new1.py')


]

# import all the arch modules
_arch_modules = [
    importlib.import_module(f'basicsr.models.archs.{file_name}')
    for file_name in arch_filenames
]


def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class: Instantiated class.
    """

    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)


def define_network(opt):
    network_type = opt.pop('type')
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net
