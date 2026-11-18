ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
FROM ${PYTHON_IMAGE} AS build
RUN python -m pip install --no-cache-dir uv==0.7.11
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY chem_repro ./chem_repro
RUN uv sync --frozen --no-dev --no-editable && uv cache clean

FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libcairo2 libxrender1 libxext6 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    CHEM_REPRO_ARTIFACTS_DIR=/artifacts \
    CHEM_REPRO_STRUCTURE_RESULTS_DIR=/runs/structure \
    CHEM_REPRO_FUSION_RESULTS_DIR=/runs/fusion \
    CHEM_REPRO_OPEN_TOX_RESULTS_DIR=/runs/open_tox \
    CHEM_REPRO_TOX_PROFILE_RESULTS_DIR=/runs/tox_profiles
RUN useradd --uid 10001 --create-home chem \
    && mkdir -p /srv /artifacts /runs /sources \
    && chown -R chem:chem /srv /artifacts /runs /sources
WORKDIR /srv
USER chem
EXPOSE 7341 7342 7343
CMD ["cannabis-2026-mcp", "--host", "0.0.0.0"]

FROM runtime AS docking
USER root
RUN /usr/local/bin/python -m pip install --no-cache-dir uv==0.7.11
COPY docker/docking-requirements.txt /tmp/docking-requirements.txt
RUN uv venv /opt/docking --python /usr/local/bin/python \
    && uv pip install --python /opt/docking/bin/python -r /tmp/docking-requirements.txt \
    && uv cache clean
COPY scripts/install_vina_binary.py /tmp/install_vina_binary.py
RUN mkdir -p /opt/bin && /opt/venv/bin/python /tmp/install_vina_binary.py /opt/bin/vina
ENV CHEM_REPRO_DOCKING_PYTHON=/opt/docking/bin/python \
    CHEM_REPRO_REDOCKING_RESULTS_DIR=/runs/heracleum_redocking_v2 \
    CHEM_REPRO_REDOCKING_SOURCES_DIR=/sources/redocking \
    CHEM_REPRO_VINA_BINARY=/opt/bin/vina
USER chem
