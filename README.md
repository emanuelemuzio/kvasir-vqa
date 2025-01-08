# KvasirVQA

La repository contiene il codice sorgente e i risultati degli esperimenti relativi alla tesi di ricerca su tecniche di VQA applicate al dataset KvasirVQA

## Struttura del progetto

La struttura del progetto è la seguente:

|```root```

|--```.vscode```

|---- ```launch.json``` _file contenente configurazioni di lancio degli script_

|--```data```

|----```augmentation``` _cartella contenente il risultato della data augmentation per l'estrattore delle feature_

|----```hyper-kvasir``` _cartella contenente dati del dataset hyper-kvasir_

|----```kvasir-instrument``` _cartella contenente dati del dataset kvasir-instrument_

|----```kvasir-vqa``` _cartella contenente le immagini del dataset kvasir-vqa e quelle generate sinteticamente dalla DA_

|----```aug_metadata.csv``` _file di metadati relativo al KvasirVQA contenente il prodotto della DA_

|----```metadata.csv``` _file di metadati relativo al KvasirVQA_

|----```prompt_metadata_aug.csv``` _contiene il testo generato dal prompt tuning sui dati aumentati_

|----```prompt_metadata.csv``` _contiene il testo generato dal prompt tuning sui dati standard_

|--```feature_extractor_aug.csv``` _file di mapping con DA per addestramento del FE_

|--```feature_extractor_classes.json``` _file contenente il mapping delle classi per il FE_

|--```feature_extractor.csv``` _file di mapping senza DA per addestramento del FE_

|--```logs``` _cartella contenente i file di log_

|--```models```

|----```vilt``` _cartella contenente dati relativi agli esperimenti con architettura ViLT_

|----```blip``` _cartella contenente dati relativi agli esperimenti con architettura BLIP_

|--```scripts``` _cartella contenente gli scripts_

|----```blip.py, custom.py vilt.py``` _scripts da lanciare per l'addestramento di architetture custom o specifiche_

|----```feature_extractor.py``` _script da lanciare per il pre addestramento del FE_

|----```generate_prompts.py``` _script per la generazione dei prompt al fine di sperimentare il prompt tuning_

|----```retrieve_kvasir_vqa.py``` _script per il recupero dei dati relativi al KvasirVQA_

|--```src``` _sorgenti e implementazioni, definizioni di dataset e funzioni di utility_

|--```.env``` _file con variabili di ambiente, fondamentale per il corretto funzionamento_

|--```kvasir_vqa.ipynb``` _Notebook contenente analisi dei dati sul KvasirVQA_

|--```feature_extractor.ipynb``` _Notebook contenente analisi dei dati per il FE_

|--```Modelfile``` _File con specifiche dell'LLM per il prompting_

## Requisiti

Le dipendenze sono specificate nel file ```requirements.txt```.

Per l'occasione è stato inoltre configurato un ambiente virtuale ```venv``` in Python.

Occorre poi lanciare i seguenti script da linea di comando:

- Per la creazione dell'ambiente virtuale:

    - ```python -m venv venv```
    - ```source venv/bin/activate``` o su Windows ```venv/Scripts/activate```

- Per l'installazione delle dipendenze:
    - ```pip install -r requirements.txt```

## Contatti

Per domande o altro contattami a:

- **Email**: emanuelemuzio@hotmail.it

python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
deactivate
ollama create kvasirvqa -f ./Modelfile 