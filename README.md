# 环境

```
python 3.12.11
Package                                  Version

---------------------------------------- ------------

accelerate                               1.10.1
aiohappyeyeballs                         2.6.1
aiohttp                                  3.13.0
aiosignal                                1.4.0
annotated-types                          0.7.0
anyio                                    4.11.0
async-timeout                            4.0.3
attrs                                    25.4.0
backoff                                  2.2.1
bcrypt                                   5.0.0
bitsandbytes                             0.48.1
Brotli                                   1.0.9
build                                    1.3.0
cachetools                               6.2.1
certifi                                  2025.10.5
cffi                                     2.0.0
charset-normalizer                       3.4.3
chromadb                                 1.2.1
click                                    8.3.0
colorama                                 0.4.6
coloredlogs                              15.0.1
cryptography                             46.0.3
dataclasses-json                         0.6.7
diffusers                                0.35.1
distro                                   1.9.0
durationpy                               0.10
einops                                   0.8.1
exceptiongroup                           1.3.0
filelock                                 3.20.0
flatbuffers                              25.9.23
frozenlist                               1.7.0
fsspec                                   2025.9.0
google-auth                              2.41.1
googleapis-common-protos                 1.71.0
greenlet                                 3.2.4
grpcio                                   1.76.0
h11                                      0.16.0
h2                                       4.3.0
hpack                                    4.1.0
httpcore                                 1.0.9
httptools                                0.7.1
httpx                                    0.28.1
httpx-sse                                0.4.3
huggingface-hub                          0.35.3
humanfriendly                            10.0
hyperframe                               6.1.0
idna                                     3.11
importlib_metadata                       8.7.0
importlib_resources                      6.5.2
Jinja2                                   3.1.6
jiter                                    0.11.1
joblib                                   1.5.2
jsonpatch                                1.33
jsonpointer                              2.0
jsonschema                               4.25.1
jsonschema-specifications                2025.9.1
kubernetes                               34.1.0
langchain                                0.0.350
langchain-classic                        1.0.0
langchain-community                      0.0.8
langchain-core                           0.1.23
langchain-openai                         1.0.1
langchain-text-splitters                 0.3.11
langgraph                                1.0.1
langgraph-checkpoint                     3.0.0
langgraph-prebuilt                       1.0.1
langgraph-sdk                            0.2.9
langsmith                                0.0.87
markdown-it-py                           4.0.0
MarkupSafe                               3.0.3
marshmallow                              3.26.1
mdurl                                    0.1.2
mkl_fft                                  1.3.11
mkl_random                               1.2.8
mkl-service                              2.5.2
mmh3                                     5.2.0
modelscope                               1.30.0
mpmath                                   1.3.0
multidict                                6.6.3
mypy_extensions                          1.1.0
mysql                                    0.0.3
mysql-connector-python                   9.5.0
mysqlclient                              2.2.7
networkx                                 3.5
numpy                                    1.26.4
oauthlib                                 3.3.1
onnxruntime                              1.23.2
openai                                   2.6.1
opentelemetry-api                        1.38.0
opentelemetry-exporter-otlp-proto-common 1.38.0
opentelemetry-exporter-otlp-proto-grpc   1.38.0
opentelemetry-proto                      1.38.0
opentelemetry-sdk                        1.38.0
opentelemetry-semantic-conventions       0.59b0
orjson                                   3.10.14
ormsgpack                                1.11.0
overrides                                7.7.0
packaging                                23.2
pandas                                   2.3.3
peft                                     0.17.1
pillow                                   11.3.0
pip                                      25.2
posthog                                  5.4.0
propcache                                0.3.1
protobuf                                 6.32.1
psutil                                   7.1.0
py-cpuinfo                               9.0.0
pyasn1                                   0.6.1
pyasn1_modules                           0.4.2
pybase64                                 1.4.2
pycparser                                2.22
pydantic                                 2.12.3
pydantic_core                            2.41.4
pydantic-settings                        2.11.0
Pygments                                 2.19.2
PyMySQL                                  1.1.2
PyPika                                   0.48.9
pyproject_hooks                          1.2.0
pyreadline3                              3.5.4
PySocks                                  1.7.1
python-dateutil                          2.9.0.post0
python-dotenv                            1.1.1
pytz                                     2025.2
PyYAML                                   6.0.3
referencing                              0.37.0
regex                                    2025.9.18
requests                                 2.32.5
requests-oauthlib                        2.0.0
requests-toolbelt                        1.0.0
rich                                     14.2.0
rpds-py                                  0.28.0
rsa                                      4.9.1
safetensors                              0.6.2
scikit-learn                             1.7.2
scipy                                    1.16.2
sentence-transformers                    5.1.2
setuptools                               80.9.0
shellingham                              1.5.4
six                                      1.17.0
sniffio                                  1.3.1
SQLAlchemy                               2.0.43
sympy                                    1.14.0
tenacity                                 8.5.0
threadpoolctl                            3.6.0
tiktoken                                 0.12.0
tokenizers                               0.22.1
torch                                    2.9.0+cu126
torchvision                              0.24.0+cu126
tqdm                                     4.67.1
transformers                             4.57.0
transformers-stream-generator            0.0.5
typer                                    0.20.0
typing_extensions                        4.15.0
typing_inspect                           0.9.0
typing-inspection                        0.4.2
tzdata                                   2025.2
urllib3                                  2.3.0
uvicorn                                  0.38.0
watchfiles                               1.1.1
websocket-client                         1.9.0
websockets                               15.0.1
wheel                                    0.45.1
win_inet_pton                            1.1.0
xxhash                                   3.5.0
yarl                                     1.22.0
zipp                                     3.23.0
zstandard                                0.23.0
```

# 运行

下载Qwen3-8b

python download.py

git clone https://github.com/QwenLM/Qwen3.git

## 创建测试数据库

cd Qwen3

cd rag_sql 

python create_mid_database.py

## 运行测试系统

cd ..

cd rag_system


python database_rag_system.py
