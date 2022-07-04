### SAT chatbot web app

#### Notes: 

1) Before running the code in this folder, please obtain the files 'RoBERTa_emotion_best.pt' and 'T5_empathy_best.pt' by running the Jupyter notebooks 'emotion classifier - RoBERTa fine-tuned on Emotion + our data.ipynb' and 'empathy classifier - T5 finetuned on our data.ipynb' in the NLP models folder

2) You may need to change the file paths in 'classifiers.py' and 'rule_based_model.py' to your local paths when running locally

3) This chatbot uses the react-chatbot-kit library: https://fredrikoseberg.github.io/react-chatbot-kit-docs/

### _Added since EmpatheticPersonas12_

1) in the Notes refers to the previous version of the chatbot for models trained only on 4 emotions. Refer to "New Notebooks" in "NLP Models" for the newly trained models on EmpatheticPersonas12

For the newly available models, based on EmpatheticPersonas12, you would want to obtain 'T5_teacher_RoBERTa_student' and 'Megatron_BERT_finetuned.pt' for to load into the 'classifiers.py' paths.

Before loading the Megatron model, you need to download a checkpoint from the nvidia repository. This checkpoint needs to be converted. Follow the steps below from the root level of this repo(based on this tutorial https://huggingface.co/nvidia/megatron-bert-uncased-345m):

```
$ git clone https://github.com/huggingface/transformers.git SATbot/model/transformers

$ mkdir -p SATbot/model/nvidia/megatron-bert-cased-345m/

$ wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O SATbot/model/nvidia/megatron-bert-cased-345m/checkpoint.zip

$ python3 transformers/src/transformers/models/megatron_bert/convert_megatron_bert_checkpoint.py SATbot/model/nvidia/megatron-bert-cased-345m/checkpoint.zip
```

You should end up with three files in the nvidia/megatron-bert-cased-345m/ folder:
```config.json, merges.txt, pytorch_model.bin```. These files can then be loaded with the ```MegatronBertModel``` and ```BertTokenizer``` classes from the standard transformers library, but navigating them to the newly downloaded files.


#### To run the code in this folder locally, after cloning open a terminal window and do:

$ pip3 install virtualenv

$ virtualenv ./SATbot

$ cd ./SATbot

$ source bin/activate

$ cd ./model

$ python3 -m pip install -r requirements.txt

$ set FLASK_APP=flask_backend_with_aws

$ flask db init

$ flask db migrate -m "testDB table"

$ flask db upgrade

$ nano .env   ---->  add DATABASE_URL="sqlite:////YOUR LOCAL PATH TO THE app.db FILE" to the .env file, save and exit

$ flask run


#### To launch the front end, open another terminal tab and do:

$ cd ./SATbot/view

$ npm i

$ npm run start

