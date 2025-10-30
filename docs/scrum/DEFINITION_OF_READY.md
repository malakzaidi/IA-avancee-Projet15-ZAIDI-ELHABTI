# 📝 Definition of Ready (DoR)

**Une User Story est "Ready" (prête à être prise) quand :**

## Checklist rapide :
```
✅ Le titre est clair : "[US-XX] Faire quelque chose"
✅ La description explique QUOI et POURQUOI
✅ Il y a des critères d'acceptation (checkboxes)
✅ Les Story Points sont estimés (1, 2, 3, 5, 8, 13)
✅ Les labels sont ajoutés (epic, priority, sprint)
✅ Tout le monde comprend ce qu'il faut faire
✅ Pas de dépendances bloquantes non résolues
✅ Assez petite pour tenir dans le sprint (max 13 points)
```

---

## 🎯 Comment vérifier si une User Story est Ready :

### 1. Ouvre l'issue sur GitHub

### 2. Pose-toi ces questions :

**❓ Est-ce que je comprends ce qu'il faut faire ?**
- Oui → ✅
- Non → Demande des clarifications en commentaire

**❓ Est-ce que je peux tester que c'est fini ?**
- Oui → ✅ (les critères d'acceptation sont testables)
- Non → Ajoute des critères mesurables

**❓ Est-ce que c'est trop gros ?**
- Non (< 13 points) → ✅
- Oui → Découpe en plusieurs User Stories plus petites

**❓ Est-ce que je peux commencer maintenant ?**
- Oui → ✅
- Non → Il manque quelque chose (dataset, API, autre task)

---

## 🚦 Workflow :

### Product Backlog → Vérifie DoR → Sprint Backlog
```
❌ Pas Ready = Reste dans Product Backlog
✅ Ready = Peut aller dans Sprint Backlog
```

---

## 📝 Format User Story recommandé :
```markdown
## User Story
As a [role]
I want [feature]
So that [benefit]

## Acceptance Criteria
- [ ] Critère 1
- [ ] Critère 2
- [ ] Critère 3

## Story Points: 5

## Labels: Epic-2, P1-High, sprint-2
```

---

## ⚡ Action si pas Ready :

1. Commente dans l'issue ce qui manque
2. Assigne à quelqu'un pour clarifier
3. Ne la prends PAS dans le sprint
4. Améliore-la pendant le Backlog Refinement

---

**Règle d'or : Ne commence jamais une User Story qui n'est pas Ready !**