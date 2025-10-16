ARG ROCM_VERSION=7.0.2
ARG PY_VERSION=3.12
ARG TORCH_VERSION=2.8.0

FROM mambaorg/micromamba:2.1.1 as micromamba

FROM rocm/pytorch:rocm${ROCM_VERSION}_ubuntu24.04_py${PY_VERSION}_pytorch_release_${TORCH_VERSION}

ARG ROCM_VERSION
ARG PY_VERSION
ARG TORCH_VERSION

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    tmux \
    clang-format \
    clangd-19 \
    libstdc++-14-dev \
    vim \
    emacs \
    zsh \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN echo "set-option -g default-command \"/bin/bash -i\"" >> /home/$USERNAME/.tmux.conf

# Create a non-root user
ARG USERNAME=devuser
ARG USER_UID=1003
ARG USER_GID=$USER_UID
ARG MAMBA_USER_ID=${USER_UID}
ARG MAMBA_USER_GID=${USER_GID}
ENV MAMBA_USER=$USERNAME
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

# Silence warning where the UID and GID are out of the usual range,
# create a non-root user.
RUN sed -i 's/^\(UID_MAX\s*\).*$/\11000000000/' /etc/login.defs && \
    sed -i 's/^\(GID_MAX\s*\).*$/\11000000000/' /etc/login.defs && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    usermod -a -G render $USERNAME

# Remove default 'ubuntu' user (UID 1000) to prevent devcontainer permission conflicts
# Ref: https://github.com/rapidsai/devcontainers/pull/373
RUN if grep ubuntu:x:1000:1000 /etc/passwd >/dev/null; then userdel -f -r ubuntu; fi

# set up fzf
RUN git clone --depth 1 https://github.com/junegunn/fzf.git /home/$USERNAME/.fzf &&  /home/$USERNAME/.fzf/install && \
    chown $USERNAME:$USERNAME /home/$USERNAME/.fzf

# Adding micromamba functionality to an existing Docker image requires copying over needed files from the micromamba image
# Refer: https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html
COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

# Switch to non-root user
USER $USERNAME
# Set home directory
WORKDIR /home/$USERNAME

# clangd
ENV PATH="/usr/lib/llvm-19/bin:$PATH"

# Create the new micromamba environment
ARG MAMBA_ENV_NAME=flashinfer-py${PY_VERSION}-torch${TORCH_VERSION}-rocm${ROCM_VERSION}
ENV MAMBA_ENV_NAME=${MAMBA_ENV_NAME}

RUN \
    sed -i 's/^#\(force_color_prompt=yes\)/\1/' /home/$USERNAME/.bashrc && \
    echo '' >> /home/$USERNAME/.bashrc && \
    echo 'eval "$(micromamba shell hook --shell bash)"' >> /home/$USERNAME/.bashrc && \
    echo "micromamba activate ${MAMBA_ENV_NAME}" >> /home/$USERNAME/.bashrc

# Create a new micromamba env and install needed packages for flashinfer development
RUN /bin/micromamba create -n ${MAMBA_ENV_NAME} python=${PY_VERSION} gtest gmock bash-completion -c conda-forge && \
    /bin/micromamba run -n ${MAMBA_ENV_NAME} pip install --no-cache-dir cmake ninja scikit-build-core setuptools-scm pre-commit pytest pytest-cov black ruff ipython

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

CMD ["/bin/bash"]
