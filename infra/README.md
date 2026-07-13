# Infrastructure de Production Souveraine LexiorGPT (Conforme Loi 25)

Ce répertoire contient les fichiers de configuration pour déployer la stack technologique de LexiorGPT en production de manière hautement sécurisée, privée et souveraine au Québec (Canada).

---

## 📂 Architecture de la Stack Swarm

La stack est conçue sous forme de micro-services conteneurisés orchestrés par **Docker Swarm** :

* **vLLM Engine** : Moteur d'inférence LLM hautement optimisé (chargement du modèle distillé ou Qwen 32B AWQ, Prefix Caching).
* **LiteLLM Proxy** : Passerelle de gouvernance (gestion des clés API, load balancing) configurée avec **Zero-Logging** pour la Loi 25.
* **Nginx** : Proxy inverse gérant la terminaison SSL, la redirection HTTP->HTTPS et l'isolation réseau des conteneurs.
* **Lexior Notebook** : Interface utilisateur web.

---

## 🔒 Conformité Loi 25 & Souveraineté des Données

Afin de répondre aux exigences d'Évaluation des Facteurs relatifs à la Vie Privée (EFVP) du Québec :

1. **Hébergement Physique Local** : Déploiement ciblé sur des serveurs physiques Bare-Metal d'OVHcloud basés dans le centre de données de **Beauharnois, Québec (QC)**.
2. **Aucun transfert de données transfrontalier** : Contrairement aux API d'OpenAI/Anthropic, aucune donnée utilisateur ou document de preuve ne quitte le territoire québécois.
3. **Zero-Logging Configuration** : La passerelle LiteLLM est configurée pour désactiver la journalisation des prompts et des complétions (`disable_logging=true`). Les données d'inférence ne sont jamais stockées sur disque.
4. **Réseau Privé Isolé (vRack)** : Les conteneurs vLLM et LiteLLM communiquent uniquement sur un réseau privé interne Docker isolé. Seul le port 443 de Nginx est exposé publiquement.
5. **Volume de données chiffré** : Les pièces de preuve et fichiers temporaires sont stockés sur des disques NVMe locaux chiffrés (LUKS) sur le serveur Bare-Metal.

---

## 🚀 Guide de Déploiement pas à pas

### Étape 1 : Phase de Validation (PoC) sur RunPod

Cette étape permet de valider le fonctionnement de la stack et les temps de réponse avec environ 100 utilisateurs actifs.

1. **Provisionnement** : Louez une machine GPU à la demande (ex. **NVIDIA A100** ou **L40S**).
2. **Initialisation de Swarm** :
   ```bash
   docker swarm init
   ```
3. **Configuration** : Copiez le fichier `.env.example` en `.env` et ajustez les variables (jetons, modèle, clés).
   ```bash
   cp .env.example .env
   # Modifiez le fichier .env
   ```
4. **Lancement de la Stack** :
   ```bash
   docker stack deploy -c docker-compose.yml lexior-poc
   ```
5. **Vérification** :
   * Vérifiez l'état des conteneurs : `docker stack ps lexior-poc`
   * Testez les performances d'inférence de vLLM (Prefix Caching).

---

### Étape 2 : Production Souveraine sur OVHcloud (Scale-GPU)

Pour le déploiement final sécurisé et permanent :

1. **Serveur Bare-Metal Scale-GPU** : Commandez un serveur dédié physique équipé d'une **NVIDIA L40S** chez OVHcloud Beauharnois (QC).
2. **Isolation vRack** : Configurez le vRack d'OVHcloud dans votre espace client pour créer un réseau privé dédié et isoler l'interface d'administration.
3. **Installation de l'OS & Chiffrement** : Installez Rocky Linux ou Ubuntu Server avec le chiffrement de disque actif (LUKS).
4. **Déploiement Docker Swarm** :
   * Installez Docker et initialisez Swarm : `docker swarm init`
   * Déployez la stack de production :
     ```bash
     docker stack deploy -c docker-compose.yml lexior-prod
     ```

---

## 📈 Croissance Horizontale (Scale-Out)

Si le nombre d'utilisateurs actifs dépasse la capacité d'un seul GPU L40S, vous pouvez faire grandir l'infrastructure horizontalement :

1. Commandez un deuxième serveur Scale-GPU chez OVHcloud Beauharnois.
2. Reliez-le au même vRack.
3. Joignez-le au cluster Swarm existant en tant que *Worker node* :
   ```bash
   # Sur le serveur principal, récupérez le token :
   docker swarm join-token worker
   # Exécutez la commande affichée sur le nouveau serveur
   ```
4. Augmentez le nombre d'instances de vLLM en une seule commande pour répartir la charge :
   ```bash
   docker service scale lexior-prod_vllm=2
   ```
   *LiteLLM redistribuera automatiquement les requêtes d'inférence en Round-Robin entre les instances disponibles.*
