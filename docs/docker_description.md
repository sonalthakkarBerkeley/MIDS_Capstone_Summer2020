# Docker package
#### Overview
Available on [dockerhub](https://hub.docker.com/repository/docker/irrigate/irrigate30) `irrigate/irrigate30`, which provides a [GIS-package-rich enviornment](https://github.com/sonalthakkarBerkeley/MIDS_Capstone_Summer2020/blob/master/dockerfile)  on Linux 18.04. 

#### Notes
1. Pull down latest container image `docker pull irrigate/irrigate30`
2. List all containers `docker image ls`
3. List running containers `docker ps`
4. List stopped containers `docker ps -a`
5. Start docker `docker run --name <container_name> -p 8888:8888 -v <VM_path>:<docker_path> -ti <image id> bash`
6. Open new terminal for running container. `docker container exec -it <container_name> /bin/bash`
7. Upon entry to docker container, authenticate GEE. `earthengine authenticate`
8. Start Jupyter Notebook in container `jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root`. For URL w/ token given, use on local computer substituting in VM's public IP address.
9. Credit to Angela Wu for creating dockerfile
