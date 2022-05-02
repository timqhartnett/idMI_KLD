Model development for prediction of interfacial Dzylashinski-Moryia interaction from spin polarized density of states

Utilizing Kulbeck-Liebler Divergence

intent is to make a bayesian linear regression model of the form idmi ~ KLD + distance, which will then be fed into an analytical skyrmion radius model (TBD)

(TBD) bayes inference will then be used to fit calculated idmi to experimental distributinos of skyrmions


Versions
gpflow = 2.5.1 (does not recognize mac tensorflow, install with removed dependancy)
tensorflow =  2.8.0 (macos)
tensorflow-probability = 0.16.0
pymc3 = (TBD, current distribution incompatible with mac M1)

FULL PACKAGE LIST

# Name                    Version                   Build  Channel
absl-py                   1.0.0                    pypi_0    pypi
appnope                   0.1.3                    pypi_0    pypi
astunparse                1.6.3                    pypi_0    pypi
attrs                     21.4.0                   pypi_0    pypi
backcall                  0.2.0                    pypi_0    pypi
bzip2                     1.0.8                h3422bc3_4    conda-forge
c-ares                    1.18.1               h3422bc3_0    conda-forge
ca-certificates           2021.10.8            h4653dfc_0    conda-forge
cached-property           1.5.2                hd8ed1ab_1    conda-forge
cached_property           1.5.2              pyha770c72_1    conda-forge
cachetools                5.0.0                    pypi_0    pypi
certifi                   2021.10.8                pypi_0    pypi
charset-normalizer        2.0.12                   pypi_0    pypi
cloudpickle               1.3.0                    pypi_0    pypi
cycler                    0.11.0                   pypi_0    pypi
debugpy                   1.6.0                    pypi_0    pypi
decorator                 5.1.1                    pypi_0    pypi
deprecated                1.2.13                   pypi_0    pypi
dm-tree                   0.1.7                    pypi_0    pypi
entrypoints               0.4                      pypi_0    pypi
flatbuffers               2.0                      pypi_0    pypi
fonttools                 4.33.3                   pypi_0    pypi
gast                      0.5.3                    pypi_0    pypi
google-auth               2.6.6                    pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
gpflow                    2.5.1                    pypi_0    pypi
grpcio                    1.45.0           py38hdbc235a_0    conda-forge
h5py                      3.6.0           nompi_py38hacf61ce_100    conda-forge
hdf5                      1.12.1          nompi_hd9dbc9e_104    conda-forge
idna                      3.3                      pypi_0    pypi
importlib-metadata        4.11.3                   pypi_0    pypi
iniconfig                 1.1.1                    pypi_0    pypi
ipykernel                 6.13.0                   pypi_0    pypi
ipython                   7.33.0                   pypi_0    pypi
jedi                      0.18.1                   pypi_0    pypi
joblib                    1.1.0                    pypi_0    pypi
jupyter-client            7.3.0                    pypi_0    pypi
jupyter-core              4.10.0                   pypi_0    pypi
keras                     2.8.0                    pypi_0    pypi
keras-preprocessing       1.1.2                    pypi_0    pypi
kiwisolver                1.4.2                    pypi_0    pypi
krb5                      1.19.3               he492e65_0    conda-forge
lark                      1.1.2                    pypi_0    pypi
libblas                   3.9.0           14_osxarm64_openblas    conda-forge
libcblas                  3.9.0           14_osxarm64_openblas    conda-forge
libclang                  14.0.1                   pypi_0    pypi
libcurl                   7.83.0               h7965298_0    conda-forge
libcxx                    13.0.1               h6a5c8ee_0    conda-forge
libedit                   3.1.20191231         hc8eb9b7_2    conda-forge
libev                     4.33                 h642e427_1    conda-forge
libffi                    3.4.2                h3422bc3_5    conda-forge
libgfortran               5.0.0.dev0      11_0_1_hf114ba7_23    conda-forge
libgfortran5              11.0.1.dev0         hf114ba7_23    conda-forge
liblapack                 3.9.0           14_osxarm64_openblas    conda-forge
libnghttp2                1.47.0               hf30690b_0    conda-forge
libopenblas               0.3.20          openmp_h2209c59_0    conda-forge
libssh2                   1.10.0               h7a5bd25_2    conda-forge
libzlib                   1.2.11            h90dfc92_1014    conda-forge
llvm-openmp               14.0.3               hd125106_0    conda-forge
markdown                  3.3.6                    pypi_0    pypi
matplotlib                3.5.1                    pypi_0    pypi
matplotlib-inline         0.1.3                    pypi_0    pypi
multipledispatch          0.6.0                    pypi_0    pypi
ncurses                   6.3                  h07bb92c_1    conda-forge
nest-asyncio              1.5.5                    pypi_0    pypi
numpy                     1.21.6           py38hf29d37f_0    conda-forge
oauthlib                  3.2.0                    pypi_0    pypi
openssl                   3.0.2                h90dfc92_1    conda-forge
opt-einsum                3.3.0                    pypi_0    pypi
packaging                 21.3                     pypi_0    pypi
pandas                    1.4.2                    pypi_0    pypi
parso                     0.8.3                    pypi_0    pypi
pexpect                   4.8.0                    pypi_0    pypi
pickleshare               0.7.5                    pypi_0    pypi
pillow                    9.1.0                    pypi_0    pypi
pip                       22.0.4             pyhd8ed1ab_0    conda-forge
pluggy                    1.0.0                    pypi_0    pypi
prompt-toolkit            3.0.29                   pypi_0    pypi
protobuf                  3.20.1                   pypi_0    pypi
psutil                    5.9.0                    pypi_0    pypi
ptyprocess                0.7.0                    pypi_0    pypi
py                        1.11.0                   pypi_0    pypi
pyasn1                    0.4.8                    pypi_0    pypi
pyasn1-modules            0.2.8                    pypi_0    pypi
pygments                  2.12.0                   pypi_0    pypi
pyparsing                 3.0.8                    pypi_0    pypi
pytest                    7.1.2                    pypi_0    pypi
python                    3.8.13          hd3575e6_0_cpython    conda-forge
python-dateutil           2.8.2                    pypi_0    pypi
python_abi                3.8                      2_cp38    conda-forge
pytz                      2022.1                   pypi_0    pypi
pyzmq                     22.3.0                   pypi_0    pypi
readline                  8.1                  hedafd6a_0    conda-forge
requests                  2.27.1                   pypi_0    pypi
requests-oauthlib         1.3.1                    pypi_0    pypi
rsa                       4.8                      pypi_0    pypi
scikit-learn              1.0.2                    pypi_0    pypi
scipy                     1.8.0                    pypi_0    pypi
setuptools                62.1.0           py38h10201cd_0    conda-forge
six                       1.16.0             pyh6c4a22f_0    conda-forge
sklearn                   0.0                      pypi_0    pypi
spyder-kernels            2.3.0                    pypi_0    pypi
sqlite                    3.38.3               h40dfcc0_0    conda-forge
tabulate                  0.8.9                    pypi_0    pypi
tensorboard               2.8.0                    pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tensorflow-deps           2.8.0                         0    apple
tensorflow-macos          2.8.0                    pypi_0    pypi
tensorflow-probability    0.16.0                   pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
tf-estimator-nightly      2.8.0.dev2021122109          pypi_0    pypi
threadpoolctl             3.1.0                    pypi_0    pypi
tk                        8.6.12               he1e0b03_0    conda-forge
tomli                     2.0.1                    pypi_0    pypi
tornado                   6.1                      pypi_0    pypi
traitlets                 5.1.1                    pypi_0    pypi
typing-extensions         4.2.0                    pypi_0    pypi
urllib3                   1.26.9                   pypi_0    pypi
wcwidth                   0.2.5                    pypi_0    pypi
werkzeug                  2.1.2                    pypi_0    pypi
wheel                     0.37.1             pyhd8ed1ab_0    conda-forge
wrapt                     1.14.0                   pypi_0    pypi
wurlitzer                 3.0.2                    pypi_0    pypi
xz                        5.2.5                h642e427_1    conda-forge
zipp                      3.8.0                    pypi_0    pypi
zlib                      1.2.11            h90dfc92_1014    conda-forge
