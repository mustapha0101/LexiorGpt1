# Phase 5 : Production Grand Public — Quantification AWQ & Déploiement Économique (32k Contexte)

Ce répertoire contient les outils nécessaires pour transformer le modèle de recherche fine-tuné en version de production optimisée pour les utilisateurs finaux de Lexior.

---

## 🚀 Pourquoi cette Phase 5 ?

1. **Économie drastique des coûts d'infrastructure** : 
   Le modèle de base FP16 nécessite un GPU haut de gamme ultra-coûteux (comme un A100 80 Go à ~$2.00/h). La version quantifiée AWQ 4-bit peut tourner de manière très performante sur un GPU grand public de type **RTX 4090 24 Go** ou **L4/L40** (~$0.30 - $0.70/h), divisant la facture par 4.
2. **Support stable d'un contexte de 32 768 tokens (32k)** :
   La VRAM nécessaire au chargement des poids du modèle est réduite de 65 Go à seulement ~18 Go. Les ~6 Go de VRAM restants sur une carte de 24 Go (ou les ~30 Go sur une L40/A100) permettent d'accueillir confortablement les KV caches étendus de 32k.
3. **Conservation du raisonnement juridique québécois** :
   Le calibrage est réalisé à l'aide de notre jeu de données juridiques en français, ce qui garantit qu'aucune dégradation du raisonnement IRAC n'est induite par la compression de précision.

---

## 🛠️ Installation des Prérequis

Pour exécuter la quantification (sur une instance GPU active de développement ou sur RunPod) :

```bash
pip install torch transformers autoawq runpod
```

---

## 📥 Guide d'exécution : Quantification AWQ

Pour quantifier le modèle fine-tuné et le publier directement sur Hugging Face :

```bash
python3 quantize_awq.py \
  --model_path "intelliwork/LexiorGpt1-merged" \
  --num_samples 128 \
  --push_to_hub \
  --repo_id "intelliwork/LexiorGpt1-merged-AWQ" \
  --hf_token "VOTRE_TOKEN_HUGGINGFACE"
```

*Le script chargera automatiquement le jeu de données de calibration locale `./data/processed/generated_a2aj_cot_reconstructed.jsonl`, appliquera le template de conversation de Qwen-2.5, réalisera le calibrage des poids et exportera le modèle.*

---

## 🚢 Déploiement en Production (RunPod Cloud)

Une fois le modèle publié, déployez le serveur d'inférence vLLM optimisé sur un GPU RTX 4090 économique :

```bash
python3 deploy_vllm_awq.py \
  --api_key "VOTRE_CLE_API_RUNPOD" \
  --hf_token "VOTRE_TOKEN_HUGGINGFACE" \
  --model_id "intelliwork/LexiorGpt1-merged-AWQ"
```

Le script configure automatiquement le conteneur vLLM pour :
* Utiliser la quantification AWQ native.
* Étendre le contexte à **32768 tokens (32k)**.
* Activer l'appel automatique d'outils (`--enable-auto-tool-choice`).
* Utiliser le parseur d'arguments compatible (`--tool-call-parser hermes`) pour l'intégration des outils MCP de Lexior.

---

## 🔒 Déploiement Souverain Local / Docker Swarm (Loi 25)

Pour faire tourner le modèle en production souveraine (OVHcloud Beauharnois) avec la stack Swarm :

1. Mettez à jour le fichier `.env` de production :
   ```env
   GEN_MODEL=intelliwork/LexiorGpt1-merged-AWQ
   ```

2. Assurez-vous que la définition du service `vllm` dans votre `docker-compose.yml` utilise la quantification et le contexte étendu :
   ```yaml
     vllm:
       image: vllm/vllm-openai:v0.5.2
       command: >
         --model ${GEN_MODEL:-intelliwork/LexiorGpt1-merged-AWQ}
         --quantization awq
         --port 8000
         --max-model-len 32768
         --gpu-memory-utilization 0.90
         --enable-prefix-caching
         --enable-auto-tool-choice
         --tool-call-parser hermes
   ```

3. Modifiez votre configuration LiteLLM (`litellm-config.yaml`) pour l'associer au modèle AWQ :
   ```yaml
   model_list:
     - model_name: intelliwork/LexiorGpt1-merged-AWQ
       litellm_params:
         model: openai/intelliwork/LexiorGpt1-merged-AWQ
         api_base: http://vllm:8000/v1
         api_key: none
   ```

4. Redéployez la stack sur votre cluster :
   ```bash
   docker stack deploy -c docker-compose.yml lexior-prod
   ```
