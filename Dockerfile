FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV OPENCV_IO_ENABLE_OPENEXR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      exiftool \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      liblensfun1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN python -c "import lpips; lpips.LPIPS(net='alex')"

COPY . /workspace

ENTRYPOINT ["bash"]
