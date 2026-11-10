"""Jeu de test du benchmark : demandes clients et arômes attendus (vérité terrain)."""

QUESTIONS = [
    {
        "id": "Q1",
        "question": "Je cherche un arôme fruité et sucré pour un yaourt à boire",
        "attendus": ["AR001", "AR004", "AR006", "AR047"],
    },
    {
        "id": "Q2",
        "question": "Pour une eau aromatisée rafraîchissante, plutôt agrume, un peu amère",
        "attendus": ["AR011", "AR013", "AR015", "AR026"],
    },
    {
        "id": "Q3",
        "question": "Je veux un goût caramel au beurre salé pour une crème glacée",
        "attendus": ["AR017"],
    },
    {
        "id": "Q4",
        "question": "Une note fumée pour des chips végétales",
        "attendus": ["AR032", "AR036"],
    },
    {
        "id": "Q5",
        "question": "Un arôme floral pour des pâtisseries orientales",
        "attendus": ["AR041", "AR042"],
    },
    {
        "id": "Q6",
        "question": "Un arôme pour un chai latte épicé",
        "attendus": ["AR031", "AR030", "AR029"],
    },
    {
        "id": "Q7",
        "question": "Une saveur umami pour un bouillon végétarien",
        "attendus": ["AR035", "AR037"],
    },
    {
        "id": "Q8",
        "question": "Un thé glacé à la pêche, très fruité",
        "attendus": ["AR047", "AR010"],
    },
]
