import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

# Assuming you have a function that outputs recommended songs
def display_recommendations(recommended_songs):
    # Convert the recommended songs list to a DataFrame for better presentation
    df = pd.DataFrame(recommended_songs)
    print("Recommendations:")
    print(df)

# Example usage of displaying recommendations
recommended_songs = [
    {"name": "Boys", "year": 2017, "artists": "Charli XCX"},
    {"name": "Falling (blackbear Remix)", "year": 2018, "artists": ["Trevor Daniel", "blackbear"]},
    {"name": "Talk", "year": 2019, "artists": ["Khalid", "Disclosure"]},
    {"name": "El Rey De Mil Coronas", "year": 2004, "artists": "Lalo Mora"},
    {"name": "Young Dumb & Broke", "year": 2017, "artists": "Khalid"},
    {"name": "Bad Idea", "year": 2018, "artists": ["pxzvc", "Shiloh Dynasty"]},
    {"name": "D'Evils", "year": 2018, "artists": "SiR"},
    {"name": "Confessions Part II", "year": 2004, "artists": "Usher"},
    {"name": "Mejor Me Alejo", "year": 2018, "artists": "Banda MS de Sergio Lizárraga"},
    {"name": "Cool Again", "year": 2018, "artists": "Shoffy"},
    {"name": "Take A Bow", "year": 2008, "artists": "Rihanna"},
    {"name": "Panama", "year": 2018, "artists": "Quinn XCII"},
    {"name": "Change the World", "year": 1999, "artists": "Eric Clapton"},
    {"name": "Breathe", "year": 2014, "artists": "Years & Years"},
    {"name": "Knee Deep (feat. Jimmy Buffett)", "year": 2010, "artists": ["Zac Brown Band", "Jimmy Buffett"]},
    {"name": "Heaven", "year": 2017, "artists": "Kane Brown"},
    {"name": "Bohemian Rhapsody - Remastered 2011", "year": 1975, "artists": "Queen"},
    {"name": "The River (Album Version)", "year": 1972, "artists": "Dan Fogelberg"},
    {"name": "Rooster - Live at the Majestic Theatre, Brooklyn, NY - April 1996", "year": 1996, "artists": "Alice In Chains"},
    {"name": "Queen Of Spades", "year": 1978, "artists": "Styx"},
    {"name": "Gethsemane (I Only Wanted To Say) - From \"Jesus Christ Superstar\" Soundtrack", "year": 1973, "artists": "Ted Neeley"},
    {"name": "Love That Burns", "year": 1968, "artists": "Fleetwood Mac"},
    {"name": "I'm in Your Care", "year": 1989, "artists": "The Canton Spirituals"},
    {"name": "Castle Walls", "year": 1977, "artists": "Styx"},
    {"name": "Hand of Doom - 2012 - Remaster", "year": 1970, "artists": "Black Sabbath"},
    {"name": "On the Low", "year": 2019, "artists": "Burna Boy"},
    {"name": "El Efecto", "year": 2019, "artists": ["Rauw Alejandro", "Chencho Corleone"]},
    {"name": "Because Of You", "year": 2007, "artists": "Ne-Yo"},
    {"name": "Nunca Es Suficiente", "year": 2018, "artists": ["Los Ángeles Azules", "Natalia Lafourcade"]},
    {"name": "Baby I'm Yours", "year": 2012, "artists": ["Breakbot", "Irfane"]},
    {"name": "Métele Sazón", "year": 2003, "artists": ["Luny Tunes", "Noriega", "Tego Calderón"]},
    {"name": "How Long", "year": 2018, "artists": "Charlie Puth"},
    {"name": "El Buho", "year": 2019, "artists": "Luis R Conriquez"},
    {"name": "Citgo", "year": 2012, "artists": "Chief Keef"},
    {"name": "deadroses", "year": 2015, "artists": "blackbear"},
    {"name": "Pink Skies (Demo)", "year": 2018, "artists": "Wiley from Atlanta"},
    {"name": "The Demo (Story)", "year": 2007, "artists": "Bs (A. Whiteman)"},
    {"name": "Grass Ain't Greener", "year": 2017, "artists": "Chris Brown"},
    {"name": "911", "year": 2018, "artists": "Ellise"},
    {"name": "Phoenix", "year": 2013, "artists": "A$AP Rocky"},
    {"name": "Rose Golden", "year": 2016, "artists": ["Kid Cudi", "WILLOW"]},
    {"name": "Kickin' Back", "year": 2016, "artists": "Mila J"},
    {"name": "Comes & Goes", "year": 2008, "artists": "Sweatshop Union"},
    {"name": "One Man's Dream", "year": 1986, "artists": "Yanni"},
    {"name": "Highschool Lover", "year": 2000, "artists": "Air"},
    {"name": "The Piano Duet", "year": 2005, "artists": "Danny Elfman"},
    {"name": "Cursum Perficio", "year": 1988, "artists": "Enya"},
    {"name": "An Evening Walk", "year": 2009, "artists": "Bernward Koch"},
    {"name": "Norrsken", "year": 2016, "artists": "Karin Borg"},
    {"name": "To Be Alone With You", "year": 2004, "artists": "Sufjan Stevens"},
    {"name": "Quidditch, Third Year", "year": 2004, "artists": "John Williams"},
    {"name": "Winter Weather / I've Got My Love to Keep Me Warm", "year": 2015, "artists": "Mason Embry Trio"},
    {"name": "Streams", "year": 2016, "artists": "Johannes Bornlöf"}
]

display_recommendations(recommended_songs)

# Assuming we have labels for ground truth and predictions
true_labels = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]  # Example ground truth labels
predicted_labels = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0]  # Example predicted labels from the recommendation system

# Calculate accuracy, precision, recall
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)

print(f"\nAccuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(conf_matrix, cmap='coolwarm')
plt.title("Confusion Matrix")
fig.colorbar(cax)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('confusion_matrix.png')

# Error Analysis - Calculating Errors
errors = np.array(true_labels) - np.array(predicted_labels)
error_count = len(errors[errors != 0])
print(f"\nNumber of Errors: {error_count}")
