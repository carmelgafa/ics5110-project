import pandas as pd


# Define the refined offence mapping
offence_mapping = {
    # Violence-related offences
    'Battery': 'Violence',
    'Felony Battery': 'Violence',
    'Batt': 'Violence',
    'Assault': 'Violence',
    'Stalking': 'Violence',
    'Abuse': 'Violence',
    'Aggress': 'Violence',
    'Threat': 'Violence',
    'Fight': 'Violence',
    'Resist': 'Violence',
    'Obstruction': 'Violence',
    'Harassment': 'Violence',
    'Intimidation': 'Violence',
    'Murder': 'Violence',
    'Homicide': 'Violence',

    # Disorder and Minor Crimes
    'Disorderly': 'Disorder',
    'Escape': 'Disorder',
    'Contempt': 'Disorder',
    'Violation': 'Disorder',
    'Arrest': 'Disorder',
    'Loiter': 'Disorder',
    'Curfew': 'Disorder',
    'Minor': 'Disorder',
    'Breach': 'Disorder',
    'Mischief': 'Disorder',

    # Theft-related offences
    'Theft': 'Theft',
    'Larceny': 'Theft',
    'Robbery': 'Theft',
    'Burglary': 'Theft',
    'Shoplifting': 'Theft',
    'Auto': 'Theft',
    'Stolen': 'Theft',
    'Grand Theft': 'Theft',
    'Petit Theft': 'Theft',

    # Fraud-related offences
    'Fraud': 'Fraud',
    'Forge': 'Fraud',
    'Forgery': 'Fraud',
    'Embezzle': 'Fraud',
    'Tamper': 'Fraud',
    'Credit': 'Fraud',
    'Identity': 'Fraud',
    'Check': 'Fraud',
    'False': 'Fraud',
    'Uttering': 'Fraud',
    'Forged': 'Fraud',
    'Misrepresent': 'Fraud',

    # Drug-related offences
    'Possession': 'Intoxication',
    'Possess': 'Intoxication',
    'Cannabis': 'Intoxication',
    'Marijuana': 'Intoxication',
    'Cocaine': 'Intoxication',
    'Heroin': 'Intoxication',
    'Meth': 'Intoxication',
    'Narcotics': 'Intoxication',
    'Drug': 'Intoxication',
    'Controlled Substance': 'Intoxication',
    'Paraphernalia': 'Intoxication',
    'Subst': 'Intoxication',
    'Contr Subst': 'Intoxication',
    'Drunk': 'Intoxication',
    'Alcoholic': 'Intoxication',
    'Intoxicated': 'Intoxication',
    'Poss' : 'Intoxication',
    'Alcohol': 'Intoxication',
    'Drinking': 'Intoxication',
    'Traffick': 'Intoxication',

    # Traffic-related offences
    'Driving': 'Traffic',
    'Drvng': 'Traffic',
    'Drivng': 'Traffic',
    'Drv': 'Traffic',
    'License': 'Traffic',
    'Lic': 'Traffic',
    'DUI': 'Traffic',
    'Susp': 'Traffic',
    'Suspended': 'Traffic',
    'Reckless': 'Traffic',
    'Speeding': 'Traffic',
    'Revoked': 'Traffic',
    'Operating': 'Traffic',
    'Vehicle': 'Traffic',
    'Veh': 'Traffic',
    'Registration': 'Traffic',
    'Expired DL': 'Traffic',

    # Morality-related offences
    'Indecent': 'Morality',
    'Loitering': 'Morality',
    'Prostitution': 'Morality',
    'Lewd': 'Morality',
    'Obscene': 'Morality',
    'Voyeur': 'Morality',
    'Exposure': 'Morality',

    # Arson-related offences
    'Arson': 'Arson',
    'Vandalism': 'Arson',
    'Trespass': 'Arson',
    'Damage': 'Arson',
    'Graffiti': 'Arson',

    # Weapon-related offences
    'Weapon': 'Weapons',
    'Firearm': 'Weapons',
    'Gun': 'Weapons',
    'Explosives': 'Weapons',
    'Knife': 'Weapons',
    'Ammo': 'Weapons',
    'Rifle': 'Weapons',


    # Domestic-related offences
    'Domestic': 'Domestic',
    'Spouse': 'Domestic',
    'Family': 'Domestic',
    'Child Abuse': 'Domestic',
    'Neglect': 'Domestic',

    # Relapse-related offences
    'Viol Pretrial': 'Relapse',
    'Pretrial Release': 'Relapse',
    'Release Dom': 'Relapse',
    'Injunc Repeat': 'Relapse',
    'Repeat Viol': 'Relapse',
    'Extradition/Defendants': 'Relapse',

    # Unresolved and other offences
    'Case No Charge': 'Unresolved',
    'Arrest Case': 'Unresolved'
}

# Function to assign categories
def assign_category(description):
    if pd.isna(description):
        return 'Other'
    for key, category in offence_mapping.items():
        if key.lower() in description.lower():
            return category
    return 'Other'
