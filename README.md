# Face to BMI
Il modello proposto in questo progetto ha lo scopo di predire il **Body Mass Index (BMI)** attraverso l'analisi di una foto del volto del soggetto. 

<p align="center">
	<img src="https://i.imgur.com/XHVvN5d.png" alt="flowchart analisi immagine"/>
</p>

## Performance
Per quanto riguarda le performance, il modello ha performato abbastanza bene durante la fase di valutazione, infatti è riuscito a raggiungere un MAE = 3,5 e un MSE = 21,49.

<p align="center">
	<img src="https://i.imgur.com/982bnYK.png" alt="metrica MSE a variare delle epoche"/>
</p>

Le prestazioni ottenute da questo modello sono migliori rispetto ai benchmark stabiliti da [Estimation of BMI from Facial Images using Semantic Segmentation based Region-Aware Pooling](https://arxiv.org/abs/2104.04733), di circa 30%, _sempre valutato rispetto al test-set usato nel progetto_.

## Dataset
Il dominio è rappresentato dal dataset in formato _csv_ con allegate immagini in formato _jpg_ scaricato da [Kaggle](https://www.kaggle.com/datasets/davidjfisher/illinois-doc-labeled-faces-dataset).  
Si tratta di un dataset formato da foto segnaletiche di detenuti e dei loro dati associati, quali: altezza, peso, ecc.  
Il copyright del dataset impiegato è di dominio pubblico poiché prodotto dal governo (_Illinois Dept. of Corrections_).  
Dal dataset originale è stato impiegato solo l’uso del data-frame _person.csv_, contenenti le informazioni per ogni soggetto, e la cartella _front_ contenenti le foto frontali di ogni soggetto.
Non è stato possibile caricare su GitHub il dataset pre e post elaborazione poiché di grandi dimensioni; per questo motivo, se si volessero recuperare i dati è possibile scaricare la cartella compressa _data_ (da estrarre e posizionare nella root del progetto) da [questo link](https://mega.nz/file/14RSHB4a#HMahTYHMI9XYLoPx55FVYnV0T7Hh55d_2jQfs7_nJrE).

## Idee progettuali
<p align="center">
	<img src="https://i.imgur.com/JqxtDMe.png" alt="workflow idee progettuali"/>
</p>

Durante la creazione del modello sono state fatte diverse prove e creati diversi modelli.
Se si volesse testare e/o visionare il codice di ciascun modello ideato, è possibile scaricare e recuperare quindi il modello corrispondente all'_idea_ (con relativo codice) da [questo link](https://mega.nz/folder/Y4ITkbaY#Zl9oZCCrTKNRT2zLWHNBRg).
Per testare dunque il modello scelto, basta spostare nella cartella _models_ presente nella cartella root del progetto il file "face_to_bmi.keras" e relativo codice sorgente trovato.
Il modello usato nel progetto finale è quello della _Idea 6_ (contrassegnato in rosso nella immagine del flowchart delle idee progettuali).

## Installazione (Windows)
1. Clonare la repository:
```
git clone https://github.com/Vincy02/Face-to-BMI
cd Face-to-BMI
```
2. Bisogna aver installato sul proprio computer due dispense di [Build Tools](https://visualstudio.microsoft.com/it/visual-cpp-build-tools/):

<p align="center">
	<img src="https://i.imgur.com/IZ18Y4K.png" alt="dispense Build Tools necessarie"/>
</p>

3. Installare i requirements per eseguire il progetto:
```
pip install -r requirements.txt
```

## Installazione (WLS con inerente supporto CUDA 12.3)
1. (Opzionale) Bisogna aver installato e configurato correttamente il supporto [CUDA 12.3](https://developer.nvidia.com/cuda-12-3-0-download-archive) e [cuDNN 8.9](https://developer.nvidia.com/rdp/cudnn-archive) per usufruire della potenza di calcolo della GPU NVIDIA.
2. Clonare la repository:
```
git clone https://github.com/Vincy02/Face-to-BMI
cd Face-to-BMI
```
3. Creazione ambiente di lavoro:
```
conda create -n face-to-bmi python=3.12.5
```
4. Attivazione ambiente di lavoro:
```
conda activate face-to-bmi
```
5. Installare CMake:
```
sudo apt-get -y install cmake
```
6. Installare librerie OS essenziali:
```
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libx11-dev libatlas-base-dev
sudo apt-get install libgtk-3-dev libboost-python-dev
sudo apt-get install libopenblas-dev liblapack-dev -y
```
7. Installare i requirements per eseguire il progetto:
```
pip install -r requirements-wsl-cuda.txt
```

## Testare il modello
1. (Opzionale) spostare nella cartella ./test l'immagine o le immagini che si vorrebbero testare.
2. Aprire con un qualsiasi editor di testo _prediction.py_ e modificare il la variabile _file_name_ con il nome della foto del volto di cui si vuole predire il BMI.
3. Runnare _prediction.py_
```
python prediction.py
```

## Disclaimer 
È fondamentale sottolineare che l'obiettivo di questo progetto è puramente accademico e di ricerca.  
L'utilizzo di un modello di questo tipo per scopi diagnostici o clinici richiederebbe ulteriori sviluppi, validazioni e regolamentazioni. Inoltre, è importante considerare le implicazioni etiche legate alla privacy e alla discriminazione nell'utilizzo di algoritmi di riconoscimento facciale.