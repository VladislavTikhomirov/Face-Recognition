crete a face_data folder in the main part of the file so that your layout looks like:


body_rec
|->body_rec≥py
|->pose_landmarker.task

face_data

face_id_model.h5
.....
object_detect
|->effiecient_lite0.tflpite
|->objectdetect_mp.py


________________________________________


where |-> is a folder indent



________________________________________




you will need to install:
________________________________________
main installs: 
keras                        3.0.0
tensorflow-macos             2.16.1
tensorflow-metal             1.1.0
________________________________________

absl-py                      2.1.0
astunparse                   1.6.3
attrs                        24.3.0
certifi                      2024.12.14
cffi                         1.17.1
charset-normalizer           3.4.1
contourpy                    1.3.1
cycler                       0.12.1
dm-tree                      0.1.8
flatbuffers                  24.12.23
fonttools                    4.55.3
gast                         0.6.0
google-pasta                 0.2.0
grpcio                       1.69.0
h5py                         3.12.1
idna                         3.10
jax                          0.4.38
jaxlib                       0.4.38
keras                        3.0.0
kiwisolver                   1.4.8
libclang                     18.1.1
Markdown                     3.7
markdown-it-py               3.0.0
MarkupSafe                   3.0.2
matplotlib                   3.10.0
mdurl                        0.1.2
mediapipe                    0.10.20
ml_dtypes                    0.5.1
namex                        0.0.8
numpy                        1.26.4
opencv-contrib-python        4.10.0.84
opencv-python                4.10.0.84
opt_einsum                   3.4.0
packaging                    24.2
pillow                       11.1.0
pip                          24.2
protobuf                     4.25.5
pycparser                    2.22
Pygments                     2.19.1
pyparsing                    3.2.1
python-dateutil              2.9.0.post0
requests                     2.32.3
rich                         13.9.4
scipy                        1.15.0
sentencepiece                0.2.0
setuptools                   74.1.2
six                          1.17.0
sounddevice                  0.5.1
tensorboard                  2.16.2
tensorboard-data-server      0.7.2
tensorflow                   2.16.1
tensorflow-io-gcs-filesystem 0.37.1
tensorflow-macos             2.16.1
tensorflow-metal             1.1.0
termcolor                    2.5.0
typing_extensions            4.12.2
urllib3                      2.3.0
Werkzeug                     3.1.3
wheel                        0.45.1
wrapt                        1.17.0



________________________________________

to run the program you need to run the ace_rec_no_mp.py file then follow the steps that are displayed on the terminal. you will need to first collect face data then train the model then run the model. a seperate run is needed per each opperation
you can add more photos to data that has already been created eg: if you have the name Vlad you can add photos to the folder Vlad without deleting and recreating the folder. You can also retrain the model without the need to delete and to start again
