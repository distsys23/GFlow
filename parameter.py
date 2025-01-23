"""
Code for parameters of VMs in hybrid cloud and the training process.
"""
from collections import namedtuple
Args = namedtuple('Args',[
    'task_num', 'VM_num', 'private_cloud_num', 'public_cloud_num', 'vm_cost', 'vm_speed', 'vm_type', 
    'privacy_factor', 'vm_security_authentication', 'vm_security_confidentiality', 'vm_security_integrity'
])

args = Args(
    task_num = 30,
    VM_num = 12,
    private_cloud_num = 6,
    public_cloud_num = 6,

    # VM=14
    # vm_cost = [1, 1, 1, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5, 1.5],
    # vm_speed = [1, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 3.5],
    # vm_type = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    # privacy_factor = 0.5,
    # vm_security_authentication = [0.93, 1, 0.734, 1, 0.786, 1, 0.912, 1, 0.756, 1, 0.623, 1, 0.874, 1],
    # vm_security_confidentiality = [0.89, 1, 0.891, 1, 0.823, 1, 0.701, 1, 0.764, 1, 0.832, 1, 0.693, 1],
    # vm_security_integrity = [0.83, 1, 0.782, 1, 0.756, 1, 0.854, 1, 0.934, 1, 0.876, 1, 0.767, 1],

    # VM=12
    vm_cost = [1, 1, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5],
    vm_speed = [1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5],
    vm_type = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    privacy_factor = 0.5,
    vm_security_authentication = [1, 0.734, 1, 0.786, 1, 0.912, 1, 0.756, 1, 0.623, 1, 0.874],
    vm_security_confidentiality = [1, 0.891, 1, 0.823, 1, 0.701, 1, 0.764, 1, 0.832, 1, 0.693],
    vm_security_integrity = [1, 0.782, 1, 0.756, 1, 0.854, 1, 0.934, 1, 0.876, 1, 0.767],

    # vm=10
    # vm_cost = [1, 1, 1.1, 1.1, 1.2, 1.3, 1.4, 1.4, 1.5, 1.5],
    # vm_speed = [1, 1, 1.5, 1.5, 2, 2.5, 3, 3, 3.5, 3.5],
    # vm_type = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    # privacy_factor = 0.5,
    # vm_security_authentication = [1, 0.734, 1, 0.786, 1, 0.756, 1, 0.623, 1, 0.874],
    # vm_security_confidentiality = [1, 0.891, 1, 0.823, 1,  0.764, 1, 0.832, 1, 0.693],
    # vm_security_integrity = [1, 0.782, 1, 0.756, 1,  0.934, 1, 0.876, 1, 0.767],

    #    vm=8
    # vm_cost = [1, 1, 1.1,  1.2, 1.3, 1.4,  1.5, 1.5],
    # vm_speed = [1, 1, 1.5,  2, 2.5, 3,  3.5, 3.5],
    # vm_type = [0, 1, 0,  0, 1,  1, 0, 1],
    # privacy_factor = 0.5,
    # vm_security_authentication = [1, 0.734, 1,  1, 0.756,  0.623, 1, 0.874],
    # vm_security_confidentiality = [1, 0.891, 1, 1,  0.764,  0.832, 1, 0.693],
    # vm_security_integrity = [1, 0.782, 1, 1,  0.934, 0.876, 1, 0.767],

    # vm = 6
    # vm_cost = [ 1, 1.1,  1.2, 1.3, 1.4,  1.5],
    # vm_speed = [1, 1.5,  2, 2.5, 3,  3.5],
    # vm_type = [ 1, 0,  0, 1,  1, 0],
    # privacy_factor = 0.5,
    # vm_security_authentication = [ 0.734, 1,  1, 0.956,  0.623, 1],
    # vm_security_confidentiality = [ 0.891, 1, 1,  0.764,  0.832, 1],
    # vm_security_integrity = [ 0.782, 1, 1,  0.934, 0.876, 1],



    # vm_cost = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5],
    # vm_speed = [1, 1.5, 2, 2.5, 3, 3.5, 1, 1.5, 2, 2.5, 3, 3.5],
    # vm_type = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # 0表示私有云 1表示公有云
    # privacy_factor = 0.5,
    # vm_security_authentication = [1, 1, 1, 1, 1, 1, 0.734, 0.786, 0.912, 0.756, 0.623, 0.874],  # 1表示 对于私有云来说，security肯定满足
    # vm_security_confidentiality = [1, 1, 1, 1, 1, 1, 0.891, 0.823, 0.601, 0.764, 0.632, 0.693],
    # vm_security_integrity = [1, 1, 1, 1, 1, 1, 0.682, 0.756, 0.654, 0.934, 0.876, 0.767],
)


