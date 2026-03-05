def plot_sector_allocation(allocation):

    sectors = {
        "ENGI.PA": "Énergie",
        "BNP.PA": "Finance",
        "ACA.PA": "Finance",
        "GLE.PA": "Finance",
        "TTE.PA": "Énergie",
        "MC.PA": "Luxe",
        "OR.PA": "Consommation",
        "AIR.PA": "Aéronautique",
        "RNO.PA": "Automobile",
        "VK.PA": "Industrie",

        "KER.PA": "Luxe",
        "RMS.PA": "Luxe",

        "SAF.PA": "Aéronautique",
        "HO.PA": "Défense",

        "SU.PA": "Industrie",
        "CAP.PA": "Technologie",
        "STMPA.PA": "Technologie",

        "EDF.PA": "Énergie",
        "VIE.PA": "Environnement",
        "EN.PA": "Construction",

        "SAN.PA": "Santé",

        "SGO.PA": "Matériaux",

        "ORA.PA": "Télécom",

        "CA.PA": "Distribution",
        "RI.PA": "Consommation",

        "DG.PA": "Construction",
        "AI.PA": "Industrie"
    }

    sector_allocation = allocation.copy()

    sector_allocation["Secteur"] = sector_allocation.index.map(sectors)

    sector_allocation["Secteur"] = sector_allocation["Secteur"].fillna("Autre")

    sector_allocation = sector_allocation.groupby("Secteur")["Poids"].sum()

    st.bar_chart(sector_allocation)
