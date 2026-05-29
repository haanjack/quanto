FROM nvcr.io/nvidia/pytorch:25.05-py3

WORKDIR /workspace/quanto

COPY . /workspace/quanto

RUN pip install -e contribs/quark
RUN pip install -e ".[dev]"

RUN pip install \
    lm-eval \
    tiktoken \
    zstandard \
    pydantic \
    onnx onnxslim onnxscript \
    scikit-learn \
    pyyaml \
    tensorboard

ENV PYTHONPATH=/workspace/quanto/src
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HF_HOME=/workspace/.cache/huggingface

CMD ["/bin/bash"]
