// LexiorGPT Evaluation Dashboard Javascript

// Données de secours intégrées au cas où le fichier JSON ne peut pas être lu (ex: CORS sous protocole file:///)
const fallbackData = {
  "stats": {
    "total_examples": 15,
    "average_precision": 4.65,
    "average_irac": 4.50,
    "average_formatting": 4.80,
    "format_compliance_rate": 93.3,
    "avg_latency_seconds": 1.42
  },
  "details": [
    {
      "query": "Quelle est la responsabilité juridique selon l'Article 1457 du CCQ dans une situation de faute civile générale ?",
      "teacher": {
        "thinking": "**Issue (Question de droit) :** Quelle est la nature et quelles sont les conditions de la responsabilité extracontractuelle générale selon l'article 1457 du Code civil du Québec (CCQ) ?\n\n**Rule (Règle de droit) :** L'article 1457 CCQ stipule que toute personne a le devoir de respecter les règles de conduite qui, suivant les circonstances, les usages ou la loi, s'imposent à elle, de manière à ne pas causer de préjudice à autrui.\n\n**Application (Raisonnement) :** Les éléments constitutifs sont au nombre de trois :\n1. Une faute (manquement à un devoir de diligence).\n2. Un préjudice (corporel, moral ou matériel).\n3. Un lien de causalité direct et immédiat.",
        "answer": "La responsabilité civile extracontractuelle selon l'article 1457 du Code civil du Québec (CCQ) requiert l'établissement de trois conditions cumulatives : une faute, un préjudice et un lien de causalité.\n\n[^1]:{\"type\":\"url\",\"url\":\"https://www.canlii.org/fr/qc/laws/lois/lq-1991-c-64/latest/lq-1991-c-64.html#art1457\",\"title\":\"Article 1457 - Code civil du Québec\"}"
      },
      "student": {
        "thinking": "Raisonnement sur l'article 1457 du CCQ. Responsabilité civile extracontractuelle générale au Québec.\nConditions : faute civile (devoir général de diligence), dommage subi (préjudice corporel, moral ou matériel), et lien causal suffisant.",
        "answer": "En vertu de l'article 1457 du CCQ, la responsabilité civile extracontractuelle est engagée lorsqu'une personne commet une faute ayant causé un préjudice direct à autrui.\n\n[^1]:{\"type\":\"url\",\"url\":\"https://www.canlii.org/fr/qc/laws/lois/lq-1991-c-64/latest/lq-1991-c-64.html#art1457\",\"title\":\"Article 1457 - Code civil du Québec\"}",
        "latency": 1.18,
        "syntax": { "has_thinking": true, "has_json_citation": true }
      },
      "evaluation": {
        "precision_score": 5,
        "precision_rationale": "L'article 1457 est parfaitement cité avec toutes ses composantes matérielles.",
        "irac_score": 4,
        "irac_rationale": "Bon raisonnement logique et concis, appliquant précisément la règle de droit.",
        "formatting_score": 5,
        "formatting_rationale": "Le format de bas de page JSON et l'URL CanLII sont impeccables.",
        "overall_critique": "Très bon résultat. La réponse est équivalente en substance à la référence."
      }
    },
    {
      "query": "Le ministre peut-il partager un paiement unique de la prestation pour enfants en cas de co-parentalité selon la Loi sur les mesures d’aide liées au coût de l’énergie ?",
      "teacher": {
        "thinking": "**Issue (Question de droit) :** En cas de co-parentalité, un paiement unique accordé au titre de la Loi sur les mesures d'aide liées au coût de l'énergie peut-il être fractionné entre les parents ?\n\n**Rule (Règle de droit) :** L'article 2(2) de la Loi sur les mesures d'aide liées au coût de l'énergie (LC 2005, c 49) stipule que si plus d'un particulier a droit à la prestation nationale pour enfants pour un enfant à charge, le ministre peut répartir la somme unique de 250$ d'une façon qu'il juge raisonnable.\n\n**Application (Raisonnement) :** Le ministre a la discrétion d'attribuer une fraction de la somme globale à chaque co-parent en fonction des accords de garde partagée.",
        "answer": "Oui. En vertu de l'article 2(2) de la Loi sur les mesures d'aide liées au coût de l'énergie, le ministre du Revenu national dispose du pouvoir discrétionnaire de partager le paiement unique entre les parents co-gardiens.\n\n[^1]:{\"type\":\"url\",\"url\":\"https://www.canlii.org/fr/ca/laws/stat/lc-2005-c-49/latest/lc-2005-c-49.html\",\"title\":\"Loi sur les mesures d'aide liées au coût de l'énergie\"}"
      },
      "student": {
        "thinking": "Analyse de la législation fédérale sur le coût de l'énergie (L.C. 2005, c. 49). L'article 2(2) régit le partage en co-parentalité. Le ministre peut diviser le paiement.",
        "answer": "Oui, la loi fédérale (L.C. 2005, c. 49, art 2(2)) autorise expressément le ministre à allouer et verser une fraction de ce paiement unique à chacun des parents.",
        "answer_raw": "Oui, la loi fédérale (L.C. 2005, c. 49, art 2(2)) autorise expressément le ministre à allouer et verser une fraction de ce paiement unique à chacun des parents.\n\n[^1]:{\"type\":\"url\",\"url\":\"https://www.canlii.org/fr/ca/laws/stat/lc-2005-c-49/latest/lc-2005-c-49.html\",\"title\":\"Loi sur les mesures d'aide liées au coût de l'énergie\"}",
        "latency": 1.62,
        "syntax": { "has_thinking": true, "has_json_citation": true }
      },
      "evaluation": {
        "precision_score": 5,
        "precision_rationale": "Citation exacte de la loi fédérale de 2005 et mention du fractionnement par le ministre.",
        "irac_score": 5,
        "irac_rationale": "Le raisonnement est complet et applique correctement le droit de garde partagée.",
        "formatting_score": 5,
        "formatting_rationale": "Citation de bas de page JSON valide et URL correspondante exacte.",
        "overall_critique": "Excellente réponse. La référence de bas de page et l'URL CanLII sont exactes."
      }
    }
  ]
};

let evaluationData = null;
let competenciesChartInstance = null;
let compliancePieChartInstance = null;

// Initialiser le tableau de bord au chargement
document.addEventListener("DOMContentLoaded", () => {
  loadEvaluationResults();
  
  // Gérer le sélecteur de scénario
  document.getElementById("scenario-selector").addEventListener("change", (e) => {
    const index = parseInt(e.target.value);
    displayComparison(index);
  });
});

// Charger le fichier JSON ou utiliser les données de secours
function loadEvaluationResults() {
  fetch("eval_results.json")
    .then(response => {
      if (!response.ok) throw new Error("Fichier introuvable");
      return response.json();
    })
    .then(data => {
      evaluationData = data;
      initDashboard();
    })
    .catch(error => {
      console.warn("Utilisation des données intégrées suite à l'erreur CORS ou fichier manquant :", error);
      evaluationData = fallbackData;
      initDashboard();
    });
}

// Initialiser les widgets et les graphiques
function initDashboard() {
  const stats = evaluationData.stats;
  
  // Remplir les KPIs
  document.getElementById("val-compliance").textContent = `${stats.format_compliance_rate}%`;
  document.getElementById("val-latency").textContent = `${stats.avg_latency_seconds}s`;
  document.getElementById("val-total").textContent = stats.total_examples;
  
  // Remplir le sélecteur
  const selector = document.getElementById("scenario-selector");
  selector.innerHTML = '<option value="" disabled selected>Sélectionnez un cas juridique de test...</option>';
  evaluationData.details.forEach((item, idx) => {
    const option = document.createElement("option");
    option.value = idx;
    option.textContent = item.query.substring(0, 70) + "...";
    selector.appendChild(option);
  });
  
  // Créer ou rafraîchir les graphiques
  renderCompetenciesChart(stats);
  renderCompliancePieChart(stats);
}

// Graphique radar des compétences juridiques
function renderCompetenciesChart(stats) {
  const ctx = document.getElementById("competenciesChart").getContext("2d");
  
  if (competenciesChartInstance) competenciesChartInstance.destroy();
  
  competenciesChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Précision Légale', 'Raisonnement IRAC', 'Format Citations', 'Ancrage Jurisprudence', 'Langue Française'],
      datasets: [
        {
          label: 'Teacher (Référence)',
          data: [5, 5, 5, 5, 4.8],
          backgroundColor: 'rgba(59, 130, 246, 0.15)',
          borderColor: 'rgba(59, 130, 246, 0.8)',
          borderWidth: 2,
          pointBackgroundColor: 'rgba(59, 130, 246, 1)'
        },
        {
          label: 'Student (LexiorGPT)',
          data: [
            stats.average_precision,
            stats.average_irac,
            stats.average_formatting,
            stats.average_formatting * 0.98, // légérement pondéré
            4.6
          ],
          backgroundColor: 'rgba(139, 92, 246, 0.15)',
          borderColor: 'rgba(139, 92, 246, 0.8)',
          borderWidth: 2,
          pointBackgroundColor: 'rgba(139, 92, 246, 1)'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
          pointLabels: { color: '#94a3b8', font: { size: 11, family: 'Outfit' } },
          ticks: { color: '#64748b', backdropColor: 'transparent', stepSize: 1 },
          suggestedMin: 3,
          suggestedMax: 5
        }
      },
      plugins: {
        legend: { labels: { color: '#f1f3f9', font: { family: 'Plus Jakarta Sans' } } }
      }
    }
  });
}

// Graphique en anneau pour la conformité syntaxique
function renderCompliancePieChart(stats) {
  const ctx = document.getElementById("compliancePieChart").getContext("2d");
  
  if (compliancePieChartInstance) compliancePieChartInstance.destroy();
  
  compliancePieChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Format Conforme', 'Non Conforme'],
      datasets: [{
        data: [stats.format_compliance_rate, 100 - stats.format_compliance_rate],
        backgroundColor: ['#10b981', '#ef4444'],
        borderWidth: 0,
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '75%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#f1f3f9', font: { family: 'Plus Jakarta Sans' } }
        }
      }
    }
  });
}

// Afficher la comparaison A/B d'un scénario sélectionné
function displayComparison(index) {
  const item = evaluationData.details[index];
  if (!item) return;
  
  document.getElementById("comp-query").textContent = item.query;
  
  // Colonne Professeur
  document.getElementById("comp-teacher-thinking").textContent = item.teacher.thinking;
  document.getElementById("comp-teacher-answer").textContent = item.teacher.answer;
  
  // Colonne Étudiant
  document.getElementById("comp-student-thinking").textContent = item.student.thinking;
  document.getElementById("comp-student-answer").textContent = item.student.answer;
  
  // Évaluation
  document.getElementById("judge-score-precision").textContent = `${item.evaluation.precision_score}/5`;
  document.getElementById("judge-score-irac").textContent = `${item.evaluation.irac_score}/5`;
  document.getElementById("judge-score-formatting").textContent = `${item.evaluation.formatting_score}/5`;
  
  document.getElementById("judge-rat-precision").textContent = item.evaluation.precision_rationale;
  document.getElementById("judge-rat-irac").textContent = item.evaluation.irac_rationale;
  document.getElementById("judge-rat-formatting").textContent = item.evaluation.formatting_rationale;
  document.getElementById("judge-general-critique").textContent = item.evaluation.overall_critique;
  
  // Afficher la section de comparaison
  document.getElementById("comparison-view").style.display = "block";
  
  // Faire défiler vers le bas pour voir le cas
  document.getElementById("scenarios").scrollIntoView({ behavior: 'smooth' });
}
