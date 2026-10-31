# BugsInPy++ (UNIST-LOFT/BugsInPyPP) on the image its authors verified 481 bugs
# with, plus what the orchestrator needs. linux/amd64 because several bugs pin
# Python 3.5-3.7 and TensorFlow builds that conda only ships for x86-64.
FROM --platform=linux/amd64 docker.io/continuumio/miniconda3:23.3.1-0

# Debian 11 is past LTS and its live mirrors now 404 on packages the indexes still
# list. A snapshot from the image's release month matches its installed libc6
# (2.31-13+deb11u5) and keeps the toolchain reproducible.
RUN printf '%s\n' \
      'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20230501T000000Z bullseye main' \
      'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20230501T000000Z bullseye-updates main' \
      'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20230501T000000Z bullseye-security main' \
      > /etc/apt/sources.list \
 && rm -f /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources \
 && apt-get update \
 && apt-get -y install git nano dos2unix build-essential \
 && rm -rf /var/lib/apt/lists/*

ARG BUGSINPY_COMMIT=bbb7d38a99446af8471ae205db4aecb4a1e2bdb4
RUN git clone https://github.com/UNIST-LOFT/BugsInPyPP.git /home/user/BugsInPy \
 && git -C /home/user/BugsInPy checkout --quiet "$BUGSINPY_COMMIT"

# The base env's Python 3.10 runs llmorpheus (stdlib only) and the Excel export.
RUN /opt/conda/bin/pip install --no-cache-dir openpyxl==3.1.5

# No PYTHONPATH here: it would leak into every bug's tests. run_bugs.py sets it
# only for the llmorpheus process.
ENV PATH=/home/user/BugsInPy/framework/bin:$PATH \
    CONDA_PATH=/opt/conda
WORKDIR /work
