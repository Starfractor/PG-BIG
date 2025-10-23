FROM nvidia/cuda:12.2.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc || true

WORKDIR /PG-BIG

# System Setup
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget git ca-certificates bzip2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda 
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh && \
    bash Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN /opt/conda/bin/conda init bash

# Set default environment
ENV CONDA_DEFAULT_ENV=PG-BIG
ENV PATH="/opt/conda/envs/PG-BIG/bin:/opt/conda/bin:$PATH"

RUN docker system prune -af
RUN docker volume prune -f

# Copy environment file and create conda environment
COPY environment.yaml .
RUN conda env create -f environment.yaml && \
    conda clean -afy && \
    rm -rf /root/.cache/pip

# Copy environment files and create conda environments
COPY environment.yaml opensim-processing.yaml .
RUN conda env create -f environment.yaml && \
    conda env create -f opensim-processing.yaml && \
    conda clean -afy && \
    rm -rf /root/.cache/pip

# Set environment variables for conda
SHELL ["conda", "run", "-n", "PG-BIG", "/bin/bash", "-c"]

# Copy repository
COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "PG-BIG", "python"]