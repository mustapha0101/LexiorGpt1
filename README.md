# Lexior - Distillation CoT Juridique en 2 Phases

Ce projet implémente un pipeline professionnel en deux phases pour distiller le raisonnement d'un grand modèle de langage (**Teacher**) vers un modèle de langage local plus petit et performant (**Student**), spécialisé dans le droit canadien et québécois en utilisant la structure de raisonnement **IRAC (Issue, Rule, Application, Conclusion)**.

Le modèle local apprend à écrire son raisonnement à l'intérieur de balises `<thinking>...</thinking>` avant de générer la conclusion finale en français.

---

## 📂 Organisation du Projet

Le projet est structuré en deux phases indépendantes pour une meilleure compréhension et séparation des ressources :

```text
DistillationModeles/
├── setup_env.sh                        # Configuration globale des clés d'API (HF, RunPod, OpenAI, WandB)
├── README.md                           # Documentation générale du projet
│
├── Phase1_Data_Preparation/            # PHASE 1 : Préparation & Génération (Machine locale ou CPU Pod)
│   ├── generator_a2aj.py               # Générateur parallèle (Multithreading) de CoT à partir des bases A2AJ
│   ├── dataset_formatter.py            # Formatage du dataset au template de chat Llama-3/Qwen
│   ├── dataset_eda.ipynb               # Analyse exploratoire de la qualité du dataset généré
│   └── run_generation.sh               # Script d'exécution maître pour la Phase 1
│
└── Phase2_FineTuning/                  # PHASE 2 : Entraînement GPU (RunPod GPU)
    ├── train_unsloth.py                # Script d'entraînement QLoRA 4-bit optimisé Unsloth
    ├── deploy_runpod.py                # Script d'orchestration pour louer/déployer le GPU sur RunPod
    ├── distillation_notebook.ipynb     # Notebook interactif pas à pas du fine-tuning
    └── run_training.sh                 # Script d'exécution maître pour la Phase 2 (lancé sur le GPU)
```

---

## 🛠️ Phase 1 : Préparation des Données

Cette phase s'exécute sur votre machine locale ou sur une instance CPU économique. Elle permet d'extraire la matière première des bases brutes de droit canadien/québécois de l'A2AJ et de générer un dataset CoT structuré.

### Étape 1.1 : Configurer vos clés d'API
Éditez le fichier `setup_env.sh` à la racine pour ajouter vos clés :
```bash
source setup_env.sh
```

### Étape 1.2 : Lancer la génération et le formatage
Lancez le script maître de la Phase 1 :
```bash
cd Phase1_Data_Preparation
./run_generation.sh
```
*Note : Vous pouvez modifier les variables d'environnement dans le script pour changer le nombre de documents à générer (`GEN_LIMIT`), le nombre de threads d'appels API parallèles (`GEN_WORKERS`) ou le modèle Teacher (`GEN_MODEL`).*

Les fichiers générés et formatés seront sauvegardés dans `Phase1_Data_Preparation/data/processed/`.

---

## 🚀 Phase 2 : Fine-Tuning avec Unsloth (RunPod GPU)

Cette phase nécessite un GPU NVIDIA performant avec 24 Go de VRAM (ex: **RTX 3090** ou **RTX 4090**).

### Option A : Déploiement automatique depuis votre machine locale (Recommandé)
Vous pouvez orchestrer la création du GPU et lancer l'entraînement à distance depuis votre terminal local :

1. Installez le SDK RunPod en local :
   ```bash
   pip install runpod
   ```
2. Chargez vos variables d'environnement :
   ```bash
   source setup_env.sh
   ```
3. Exécutez le script d'orchestration en passant l'URL de votre dépôt Git public ou privé :
   ```bash
   python3 Phase2_FineTuning/deploy_runpod.py \
       --gpu_type "NVIDIA GeForce RTX 3090" \
       --git_repo "https://github.com/mustapha0101/LexiorGpt1.git"
   ```

Le script va créer l'instance, cloner votre dépôt Git, lancer l'entraînement de la Phase 2 en arrière-plan, puis s'arrêter. Vous pouvez suivre l'avancement via votre [Console RunPod](https://www.runpod.io/console/pods).

### Option B : Lancement manuel sur le GPU RunPod
Si vous préférez louer l'instance manuellement via l'interface web de RunPod :

1. Déployez un GPU avec l'image `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`.
2. Connectez-vous en SSH ou ouvrez le terminal Jupyter Lab.
3. Clonez le dépôt et configurez vos variables :
   ```bash
   git clone https://github.com/mustapha0101/LexiorGpt1.git
   cd LexiorGpt1
   source setup_env.sh
   ```
4. Lancez le script d'entraînement :
   ```bash
   cd Phase2_FineTuning
   ./run_training.sh
   ```

---

## 📦 Exportation et Utilisation Locale (Ollama)

Une fois le fine-tuning terminé, le script exporte automatiquement les adaptateurs LoRA, un modèle fusionné 16-bit complet, et compile le modèle au format quantifié **GGUF** (`q4_k_m`) sur votre Hugging Face Hub (si vous l'avez configuré).

Pour exécuter votre modèle LexiorGPT localement dans **Ollama** :

1. Téléchargez le fichier `.gguf` généré.
2. Créez un fichier nommé `Modelfile` :
   ```dockerfile
   FROM ./lexiorgpt-q4_k_m.gguf

   # Configuration du prompt système orienté droit canadien
   SYSTEM """
   Tu es Lexior, un assistant d'intelligence juridique spécialisé en droit canadien et québécois.
   Tu dois formater ton raisonnement juridique complet à l'aide de la méthode IRAC dans des balises <thinking>...</thinking> avant de fournir ta réponse finale en français.
   """

   # Configuration des paramètres de créativité
   PARAMETER temperature 0.3
   PARAMETER stop "<|im_end|>"
   ```
3. Créez et lancez le modèle dans Ollama :
   ```bash
   ollama create lexiorgpt -f Modelfile
   ollama run lexiorgpt
   ```
