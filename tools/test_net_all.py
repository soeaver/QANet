from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed testing launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")
    # Optional arguments for the launch helper
    parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5,6,7', help='gpu id for test')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfgs/rcnn/mscoco/e2e_faster_rcnn_R-50-FPN_1x.yaml', type=str)
    # rest from the testing program
    parser.add_argument('testing_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    num_gpus = len(args.gpu_id.split(','))
    dist_world_size = num_gpus

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    current_env["MASTER_ADDR"] = "127.0.0.1"
    current_env["MASTER_PORT"] = "{}".format(_find_free_port())
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and num_gpus > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        # print("*****************************************\n"
        #       "Setting OMP_NUM_THREADS environment variable for each process "
        #       "to be {} in default, to avoid your system being overloaded, "
        #       "please further tune the variable for optimal performance in "
        #       "your application as needed. \n"
        #       "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    testing_script = 'tools/instance/test_net.py'

    for local_rank in range(0, num_gpus):
        # each process's rank
        dist_rank = local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        cmd = [sys.executable, "-u", testing_script, "--local_rank={}".format(local_rank), '--cfg', args.cfg_file]
        cmd.extend(args.testing_script_args)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)


if __name__ == "__main__":
    main()