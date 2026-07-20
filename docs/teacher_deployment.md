# Déploiement du Teacher vLLM

Le pilote utilise un seul pod GPU servant
`Qwen/Qwen2.5-32B-Instruct-AWQ`. L'orchestrateur reste local/CPU.

```bash
python Phase1_Data_Preparation/deploy_teacher_runpod.py
```

Le script ne lance aucune génération. Il attend une réponse HTTP valide de
`/v1/models`, avec timeout et retries, avant d'afficher les variables utiles :

```bash
export TEACHER_BASE_URL="https://<pod>-8000.proxy.runpod.net/v1"
export TEACHER_MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
export TEACHER_API_KEY="<VLLM_API_KEY, ou valeur factice sans authentification>"
```

Le SDK OpenAI n'est ici qu'un client du protocole compatible OpenAI. Avec cette
URL RunPod, aucune requête ne part vers OpenAI et aucune vraie clé OpenAI n'est
nécessaire. Si vLLM impose une clé, configurez `VLLM_API_KEY`; sinon une valeur
client factice suffit.

OpenAI reste un fournisseur optionnel : définir une URL `api.openai.com`, une
clé réelle et un modèle explicitement choisi. Aucun modèle OpenAI n'est codé en
dur. Le Critic peut employer un endpoint distinct via `CRITIC_*`.

`deploy_dual_pods.py` est legacy. Sans `--deploy-coordinator`, il ne crée plus
le second pod GPU. `deploy_generation_runpod.py` ne lance le pipeline legacy
qu'avec `--start-generation`.

En production, vLLM n'active pas le parser Hermes. Le tokenizer sauvegardé par
le trainer contient le gabarit ChatML strict; l'application hôte intercepte les
balises Lexior personnalisées.
