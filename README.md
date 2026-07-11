# Lexior - Pipeline de Distillation CoT Juridique sur RunPod

Ce dépôt contient le pipeline complet pour distiller le raisonnement d'un grand modèle de langage (**Teacher**) vers un modèle de langage local plus petit (**Student**), spécialisé dans le droit français en utilisant le format de raisonnement **IRAC (Issue, Rule, Application, Conclusion)**.

Le modèle local apprend à écrire son raisonnement à l'intérieur de balises `<thinking>...</thinking>` avant de générer la conclusion finale.

---

## 🛠️ Architecture du Pipeline

1. **Formatage (`data/dataset_formatter.py`)** : Télécharge le dataset [SuperMust/irac-thinking](https://huggingface.co/datasets/SuperMust/irac-thinking) depuis Hugging Face Hub, fusionne le raisonnement (`thinking`) et la réponse de l'assistant dans le template conversationnel de Llama 3, et génère le fichier `formatted_dataset.jsonl`.
2. **Fine-Tuning (`src/training/train_unsloth.py`)** : Entraîne le modèle local (par exemple, `Llama-3-8B` ou `Qwen-2.5-7B`) avec QLoRA en 4-bit grâce à **Unsloth** pour maximiser les performances et minimiser l'usage de la mémoire GPU (VRAM).
3. **Export et Publication** : 
   - Sauvegarde les adaptateurs LoRA.
   - Fusionne les poids pour générer un modèle complet en float16.
   - Génère automatiquement des fichiers **GGUF** (ex. `q4_k_m`) prêts pour une exécution locale rapide via **Ollama** ou **Llama.cpp**.
   - Pousse automatiquement tous les résultats sur votre Hugging Face Hub si un jeton d'accès est fourni.

---

## 🚀 Déploiement et Lancement sur RunPod

### Étape 1 : Lancement du Pod GPU
Sur [RunPod](https://www.runpod.io/), louez une instance GPU (une **RTX 3090**, **RTX 4090** ou **A40G** avec 24 Go de VRAM est idéale et économique).
* **Image Docker recommandée** : `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` ou `unslothdev/unsloth:latest`.
* Configurez au moins **40 Go de Volume Disk**.

### Étape 2 : Clonage du projet
Connectez-vous au terminal Jupyter Lab ou via SSH, puis clonez ce dépôt Git :
```bash
git clone <URL_DE_VOTRE_DEPOT_GIT>
cd DistillationModeles
```

### Étape 3 : Configuration des Variables d'Environnement
Définissez votre token Hugging Face pour l'authentification et l'URL du dépôt de destination où envoyer le modèle entraîné :
```bash
export HF_TOKEN="hf_votre_token_d_ecriture"
export HF_REPO_ID="votre_username/llama-3-8b-juridique-cot"
```

*(Optionnel)* Personnalisez les hyperparamètres et activez le suivi des expériences avec **Weights & Biases** (Wandb) :
```bash
export MODEL_NAME="unsloth/llama-3-8b-Instruct-bnb-4bit" # Ou "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
export TRAIN_EPOCHS=3
export TRAIN_BATCH_SIZE=2
export TRAIN_GRAD_ACCUM=4

# Suivi d'expériences (Weights & Biases)
export WANDB_API_KEY="votre_cle_api_wandb"
export TRACKING_REPORT_TO="wandb"
export TRACKING_RUN_NAME="llama3-juridique-r16-alpha32"
```

### Option B : Déploiement automatique par script d'orchestration (Recommandé)

Vous pouvez lancer, configurer et arrêter l'instance GPU RunPod directement depuis votre machine locale grâce au script d'orchestration `deploy_runpod.py` :

1. Installez le SDK RunPod sur votre machine locale :
   ```bash
   pip install runpod
   ```
2. Exportez vos clés d'API locales :
   ```bash
   export RUNPOD_API_KEY="votre_cle_api_runpod"
   export HF_TOKEN="votre_token_huggingface"
   export HF_REPO_ID="username/llama3-juridique-cot"
   export WANDB_API_KEY="votre_cle_api_wandb"
   ```
3. Exécutez le script en lui passant votre dépôt Git pour qu'il configure la machine, lance l'entraînement et s'éteigne (`--autostop`) automatiquement pour économiser de l'argent :
   ```bash
   python deploy_runpod.py \
       --gpu_type "NVIDIA RTX 4090" \
       --git_repo "https://github.com/votre_username/votre_depot_git.git" \
       --autostop
   ```

---

## 💻 Intégration Locale (Ollama)

Une fois l'entraînement fini, le pipeline génère un dossier contenant le modèle GGUF dans `outputs/final_model/gguf_q4_k_m/model-unsloth.gguf` (ou poussé sur Hugging Face).

Pour l'intégrer à votre application locale ou l'exécuter dans le terminal :

1. Téléchargez le fichier `.gguf`.
2. Créez un fichier nommé `Modelfile` dans le même dossier avec le contenu suivant :
   ```dockerfile
   FROM ./model-unsloth.gguf

   # Template de prompt conversationnel Llama 3
   TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

   {{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

   {{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

   {{ .Response }}<|eot_id|>"""

   # Prompt système pour forcer le raisonnement juridique IRAC
   SYSTEM """Tu es un assistant juridique francophone. Raisonne en français selon le format IRAC."""

   # Paramètres d'inférence recommandés
   PARAMETER stop "<|eot_id|>"
   PARAMETER temperature 0.3
   ```
3. Créez et lancez le modèle dans Ollama :
   ```bash
   ollama create lexior-legal-cot -f Modelfile
   ollama run lexior-legal-cot
   ```

### 🧠 Comportement lors de l'inférence
Lorsque vous posez une question juridique au modèle, sa réponse commencera toujours par :
```text
<thinking>
Issue: ...
Rule: ...
Application: ...
</thinking>

Conclusion: ...
```
L'interface de l'application de bureau **Lexior** peut capturer ce qui est produit entre les balises `<thinking>` pour l'afficher dans un volet pliable de "réflexion de l'IA" ou le masquer pour ne montrer que la conclusion juridique propre à l'utilisateur final.
