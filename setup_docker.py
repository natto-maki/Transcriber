import os
import platform
import argparse
import logging
import subprocess


_image_version = "0.3.0"

_command_template = {
    ".cpu": {
        "image": "transcriber/cpu:%(image_version)s",
        "build":
            "docker build --progress tty -t transcriber/cpu:%(image_version)s -f docker/Dockerfile.cpu "
            "--build-arg branch=%(branch)s .",
        "run":
            "docker run -it -p%(port)s:7860 "
            "--device /dev/snd:/dev/snd "
            "--name %(name)s transcriber/cpu:%(image_version)s /bin/sh -c 'cd ~/Transcriber; python3 app.py'"
    },
    ".linux_gpu": {
        "image": "transcriber/linux_gpu:%(image_version)s",
        "build":
            "docker build --progress tty -t transcriber/linux_gpu:%(image_version)s -f docker/Dockerfile.linux_gpu "
            "--build-arg branch=%(branch)s .",
        "run":
            "docker run -it --gpus %(gpus)s --shm-size=%(shm_size)s -p%(port)s:7860 "
            "--device /dev/snd:/dev/snd "
            "--name %(name)s transcriber/linux_gpu:%(image_version)s /bin/sh -c 'cd ~/Transcriber; python3 app.py'"
    },
    ".cpu.inplace": {
        "image": "transcriber/cpu.inplace:%(image_version)s",
        "build":
            "docker build --progress tty -t transcriber/cpu.inplace:%(image_version)s -f docker/Dockerfile.cpu.inplace "
            "--build-arg USER_NAME=$(id -un) --build-arg GROUP_NAME=$(id -gn) "
            "--build-arg UID=$(id -u) --build-arg GID=$(id -g) .",
        "run":
            "docker run -it --rm -p%(port)s:7860 "
            "--device /dev/snd:/dev/snd -v .:/home/$(id -un) -u $(id -u):$(id -g) --group-add=audio "
            "--name %(name)s transcriber/cpu.inplace:%(image_version)s /bin/sh -c 'python3 app.py'"
    },
    ".linux_gpu.inplace": {
        "image": "transcriber/linux_gpu.inplace:%(image_version)s",
        "build":
            "docker build --progress tty -t transcriber/linux_gpu.inplace:%(image_version)s -f docker/Dockerfile.linux_gpu.inplace "
            "--build-arg USER_NAME=$(id -un) --build-arg GROUP_NAME=$(id -gn) "
            "--build-arg UID=$(id -u) --build-arg GID=$(id -g) .",
        "run":
            "docker run -it --rm --gpus %(gpus)s --shm-size=%(shm_size)s -p%(port)s:7860 "
            "--device /dev/snd:/dev/snd -v .:/home/$(id -un) -u $(id -u):$(id -g) --group-add=audio "
            "--name %(name)s transcriber/linux_gpu.inplace:%(image_version)s /bin/sh -c 'python3 app.py'"
    }
}


def _command(args, cwd=None):
    try:
        c = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return -1, "", ""
    return c.returncode, c.stdout.decode('utf8'), c.stderr.decode('utf8')


def _docker_command(cmd, cwd=None):
    c = subprocess.run(["/bin/bash", "-c", cmd], cwd=cwd)
    return c.returncode


def main():
    parser = argparse.ArgumentParser(description="Setup docker images/containers")
    parser.add_argument("--cpu", help="force CPU mode", action="store_true", dest="cpu")
    parser.add_argument("--gpu", help="force GPU mode", action="store_true", dest="gpu")
    parser.add_argument("--inplace", help="in-place mode", action="store_true", dest="inplace")
    parser.add_argument(
        "--branch", help="branch name of the repository to clone", metavar="BRANCH_NAME",
        action="store", dest="branch", default=None)
    parser.add_argument("--build-all", help="build all images", action="store_true", dest="build_all")
    parser.add_argument(
        "--force-build", help="build image(s) even if the image is already existed",
        action="store_true", dest="force_build")
    parser.add_argument(
        "--run", help="launch docker container (only build image if not specified)",
        action="store_true", dest="run")

    gr_docker = parser.add_argument_group("Docker build/run options")
    gr_docker.add_argument("--gpus", metavar="GPU", action="store", dest="docker_gpus", default="device=0")
    gr_docker.add_argument("--shm-size", metavar="SIZE", action="store", dest="docker_shm_size", default="1g")
    gr_docker.add_argument("--port", metavar="NO", action="store", dest="docker_port", default="7860")
    gr_docker.add_argument(
        "--name", metavar="CONTAINER_NAME", action="store", dest="docker_name", default="transcriber")

    opt = parser.parse_args()

    # Find current configuration
    current_platform = "unknown"
    s = platform.system()
    if s == "Linux":
        current_platform = "Linux"
    elif s == "Windows":
        current_platform = "Windows"
    elif s == "Darwin":
        m = platform.machine()
        current_platform = "MacOS(Intel)" if m == "x86_64" else "MacOS(ARM)"

    current_gpu = "none"
    if current_platform == "Linux":
        r_code, r_stdout, _ = _command(["which", "nvidia-smi"])
        if r_code == 0 and r_stdout != "":
            current_gpu = "NVIDIA"
    if current_platform == "MacOS(ARM)":
        current_gpu = "Metal"

    may_be_inplace = False
    if os.path.isfile("emb_db.py") and os.path.isfile("llm.py"):
        may_be_inplace = True

    logging.info("current configuration: OS = %s, GPU = %s, inplace = %s" % (
        current_platform, current_gpu, "yes" if may_be_inplace else "no"))

    # Determine which Dockerfile will be met
    suffix_arch = ".linux_gpu" if current_gpu == "NVIDIA" else ".cpu"
    suffix_inplace = ".inplace" if may_be_inplace else ""

    if opt.cpu:
        suffix_arch = ".cpu"
    if opt.gpu:
        if current_gpu != "NVIDIA":
            logging.warning("NVIDIA GPU not detected, but specified to use GPU-enabled docker image")
        suffix_arch = ".linux_gpu"
    if opt.inplace:
        if not may_be_inplace:
            logging.warning("The current working directory is not the top of the repository, "
                            "but using an inplace docker image has been specified")
        suffix_inplace = ".inplace"

    branch = "main"
    if opt.branch is not None:
        suffix_inplace = ""
        branch = opt.branch

    suffix = suffix_arch + suffix_inplace
    logging.info("selected Dockerfile: docker/Dockerfile%s" % suffix)

    # Run docker commands
    kwargs = {
        "image_version": _image_version,
        "branch": branch,
        "gpus": opt.docker_gpus,
        "shm_size": opt.docker_shm_size,
        "port": opt.docker_port,
        "name": opt.docker_name,
    }

    build_targets = [suffix] if not opt.build_all else list(_command_template.keys())
    run_targets = [suffix] if opt.run else []

    # Build
    for target in build_targets:
        if not opt.force_build:
            image = _command_template[target]["image"] % kwargs
            _, r_stdout, _ = _command(["docker", "image", "ls", "-q", image])
            if len(r_stdout) != 0:
                logging.info("image %s already exist" % image)
                continue

        cmd = _command_template[target]["build"] % kwargs
        logging.info(cmd)
        _docker_command(cmd)

    # Run
    for target in run_targets:
        cmd = _command_template[target]["run"] % kwargs
        logging.info(cmd)
        _docker_command(cmd)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s: %(name)s:%(funcName)s:%(lineno)d %(levelname)s: %(message)s', level=logging.INFO)
    main()
