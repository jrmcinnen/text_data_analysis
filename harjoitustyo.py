"""
Matemaattiset ja tilastolliset ohjelmistot
Python-osio
Harjoitustyö

Tekijä:
Jere Mäkinen
jere.makinen@tuni.fi
Toukokuu 2021


Huomioita ohjelmasta:
Ohjelma toteuttaa tilastollista analyysia verkkokeskusteluaineistoon. Ohjelma toiminnot on jaettu kuuteen osaan
tehtävänantoa vastaavasti. Osat on jaettu vielä pienempiin osiin ohjelman selkeyden parantamiseksi. Ohjelma tallentaa
kuvat ja suuret taulukot työhakemistoon. Yksittäiset lukuarvot poimitaan raporttiin käsin ohjelman tulosteista.
Ohjelma on pyritty toteuttamaan siten, että vastaavanlaista analyysia voitaisiin suorittaa myös muille sanoille tai
uutisryhmille pienillä muokkauksilla alussa määriteltyihin gloobaleihin vakioihin. Oletusarvoisesti ohjelma käyttää
harjoitystyöaineistoa, mutta ohjelmaa voisi hyödyntää myös muille vastaavasti muotoiluille aineistoille.

Koodi on tehty PyCharm-kehitysympäristössä.
"""

import pandas
import statistics
import numpy
import scipy
import scipy.stats
import sklearn
import sklearn.linear_model
import math
import matplotlib.pyplot

# Eri sanoja aineistossa
WORDS_IN_DATA = 962

# Uutisryhmät ja niitä vastaavat id:t sanakirjaan talletettuina
GROUP_IDS = {'alt.atheism': 1, 'comp.graphics': 2, 'comp.os.ms-windows.misc': 3, 'comp.sys.ibm.pc.hardware': 4,
             'comp.sys.mac.hardware': 5, 'comp.windows.x': 6, 'misc.forsale': 7, 'rec.autos': 8, 'rec.motorcycles': 9,
             'rec.sport.baseball': 10, 'rec.sport.hockey': 11, 'sci.crypt': 12, 'sci.electronics': 13, 'sci.med': 14,
             'sci.space': 15, 'soc.religion.christian': 16, 'talk.politics.guns': 17, 'talk.politics.mideast': 18,
             'talk.politics.misc': 19, 'talk.religion.misc': 20}

# Sanat ja uutisryhmät, joita käsitellään osassa 1
WORDS1 = ['freedom', 'nation', 'logic', 'normal', 'program']
GROUPS1 = [['sci.crypt', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc'],
           ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc'],
           ['alt.atheism', 'sci.electronics', 'talk.politics.misc', 'talk.religion.misc'],
           ['comp.graphics', 'comp.windows.x', 'sci.electronics', 'sci.med'],
           ['comp.graphics', 'comp.windows.x', 'talk.politics.misc', 'comp.sys.mac.hardware']]


# Uutisryhmät, joita käsitellään osassa 2
GROUPS2 = [['rec.sport.baseball', 'rec.sport.hockey'], ['rec.autos', 'rec.motorcycles']]


#Uutisryhmät, joita käsitellään osassa 3
GROUPS3 = ['comp.graphics', 'soc.religion.christian']


#Sanat ja uutisryhmät, joita käsitellään osassa 4
WORDS4 = [['jpeg', 'gif'], ['write', 'sale']]
GROUPS4 = ['comp.graphics']


#Uutisryhmät, joita käsitellään osassa 5
GROUPS5 = [['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware'], ['rec.sport.baseball', 'rec.sport.hockey'],
           ['rec.autos', 'rec.motorcycles']]


#Sanat ja uutisryhmät, joita käsitellään osassa 6
WORDS6 = [['jpeg'], ['jpeg', 'earth'], ['group1'], ['group2'], ['group3'], ['group4'], ['group5'], ['group6'],
          ['group7'], ['group8']]
GROUPS6 = ['comp.graphics', 'sci.space']


# Funktio lukee harjoitustyössä käytettävän aineiston.
def read_file(file_name='harjoitustyodata.csv'):
    try:
        data = pandas.read_csv(file_name, sep=',', header='infer', quotechar='\"')
        return [data, True]
    except:
        return [[], False]


# Funktio laskee tunnuslukuja syötetylle sanalle syötetyssä uutisryhmäjoukossa.
def word_key_values(word, grouplist, data):
    key_value_dataframe = pandas.DataFrame(columns=['group', 'count', 'mean', 'median', 'SD',
                                                    '0,1% quantile', '99,9% quantile'])
    if word in data.columns:
        for group in grouplist:
            if group in GROUP_IDS:
                group_data_array = data.loc[data['groupID'] == GROUP_IDS[group]][word].to_numpy()
                count = group_data_array.sum()
                mean = group_data_array.mean()
                median = statistics.median(group_data_array)
                sd = statistics.stdev(group_data_array)
                quantiles = numpy.quantile(group_data_array, [0.001, 0.999])
                new_dataframe = pandas.DataFrame([[group, count, mean, median, sd, quantiles[0], quantiles[1]]],
                                                 columns=['group', 'count', 'mean', 'median', 'SD', '0,1% quantile',
                                                 '99,9% quantile'])
                key_value_dataframe = key_value_dataframe.append(new_dataframe, ignore_index=True)
    else:
        print("Word \"" + word + "\" is not in the data!")
        return
    key_value_dataframe.to_csv(word + "_values.csv", sep='&')
    return key_value_dataframe


#Funktio piirtää histogrammin syötetyn sanan esiintymismääristä annetussa uutisryhmäjoukossa.
def word_count_plot(word, grouplist, data):
    word_counts = []
    if word in data.columns:
        for group in grouplist:
            if group in GROUP_IDS:
                group_data_array = data.loc[data['groupID'] == GROUP_IDS[group]][word].to_numpy()
                word_counts.append(group_data_array)
            else:
                print("Group \"" + group + "\" is not in the data!")
    else:
        print("Word \"" + word + "\" is not in the data!")
        return

    word_counts = tuple(word_counts)
    fig, axes = matplotlib.pyplot.subplots()
    axes.hist(word_counts, histtype='bar')
    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.ylabel('Lukumäärä')
    matplotlib.pyplot.xlabel('Sanan esiintymiskerrat')
    matplotlib.pyplot.legend(grouplist)
    matplotlib.pyplot.title("Sanan \"" + word + "\" esiintyminen eri uutisryhmissä")
    fig.savefig("part1_hist_" + word + '.png')


# Funktio etsii aineistosta annetusta uutisryhmäjoukosta ryhmät, jossa syötettyä sanaa käytetään eniten ja vähiten
def word_usage_comparison(word, group_list, data):
    most = [-math.inf, '']
    least = [math.inf, '']
    if word in data.columns:
        for group in group_list:
            if group in GROUP_IDS:
                group_data_array = data.loc[data['groupID'] == GROUP_IDS[group]][word].to_numpy()
                count = group_data_array.sum()
                if count > most[0]:
                    most[0] = count
                    most[1] = group
                if count < least[0]:
                    least[0] = count
                    least[1] = group
            else:
                print("Group \"" + group + "\" is not in the data!")
                return
    else:
        print("Word \"" + word + "\" is not in the data!")
        return
    return most, least


# Toimenpiteet harjoitustyön osaan 1, Sanojen esiintymismäärät
def part1(data):
    for i in range(len(GROUPS1)):
        word = WORDS1[i]
        groups = GROUPS1[i]
        print("Some key values for the word \"" + word + "\":")
        print(word_key_values(word, groups, data))
        print()
        word_count_plot(word, groups, data)
        most, least = word_usage_comparison(word, groups, data)
        print('Word \"' + word + "\" was most used (" + str(int(most[0])) + " times) in the group \"" + most[1] + "\".")
        print('Word \"' + word + "\" was least used (" + str(int(least[0])) + " times) in the group \"" + least[1] +
              "\".")
        print()


# Funktio lisää aineistoon muuttujat, joihin on talletuttu vietin pituus, sekä toisen muuttujan,
# jossa on tämän pituuden logaritmi.
def add_message_lengths(data):
    lengths = []
    lengths_log = []
    for id in data['messageID']:
        message = data.loc[data['messageID'] == id]
        message_words = message.iloc[:, range(5, 5+WORDS_IN_DATA)].to_numpy()
        message_length = message_words.sum()
        lengths.append(message_length)
        lengths_log.append(math.log(message_length))
    data['length'] = lengths
    data['logarithmic length'] = lengths_log
    print("Message lengths have been added to the data")
    print(data.head(3))
    print()


# Funktio piirtää histogrammin viestin pituuksista ja viestien logaritmisista pituuksista annetussa uutisryhmäjoukossa.
def message_length_plot(grouplist, data, fig_name):
    lengths = []
    lengths_log = []
    for group in grouplist:
        if group in GROUP_IDS:
            group_data_array = data.loc[data['groupID'] == GROUP_IDS[group]]['length'].to_numpy()
            group_data_array_log = data.loc[data['groupID'] == GROUP_IDS[group]]['logarithmic length'].to_numpy()
            lengths.append(group_data_array)
            lengths_log.append(group_data_array_log)
        else:
            print("Group \"" + group + "\" is not in the data!")
            return
    lengths = tuple(lengths)
    lengths_log = tuple(lengths_log)
    fig, axes = matplotlib.pyplot.subplots()
    axes.hist(lengths, bins=30, histtype='barstacked')
    matplotlib.pyplot.legend(grouplist)
    matplotlib.pyplot.title("Viestien pituudet")
    matplotlib.pyplot.ylabel('Lukumäärä')
    matplotlib.pyplot.xlabel('Viestin pituus')
    fig.savefig("msg_length_hist" + fig_name + ".png")
    fig_log, axes_log = matplotlib.pyplot.subplots()
    axes_log.hist(lengths_log, bins=15, histtype="barstacked")
    matplotlib.pyplot.legend(grouplist)
    matplotlib.pyplot.title("Logaritmiset viestien pituudet")
    matplotlib.pyplot.ylabel('Lukumäärä')
    matplotlib.pyplot.xlabel('Logaritmi viestin pituudesta')
    fig_log.savefig("msg_length_hist_log" + fig_name + ".png")


# Funktio suorittaa odotusarvojen t-testi viestien logaritmisille pituuksille annetuille kahdelle uutisryhmälle
def ttest_message_lengths(group1, group2, data):
    if group1 in GROUP_IDS and group2 in GROUP_IDS:
        x1 = data.loc[data['groupID'] == GROUP_IDS[group1]]['logarithmic length'].to_numpy()
        x2 = data.loc[data['groupID'] == GROUP_IDS[group2]]['logarithmic length'].to_numpy()
    else:
        print("Group(s) is not in the data!")
        return
    result, pvalue = scipy.stats.ttest_ind(x1, x2)
    return result, pvalue


# Toimenpiteet harjoitustyön osaan 2, Viestien pituudet
def part2(data):
    add_message_lengths(data)
    counter = 1
    for grouplist in GROUPS2:
        message_length_plot(grouplist, data, str(counter))
        print("T-test \"" + grouplist[0] + "\" and \"" + grouplist[1] + "\":")
        print(ttest_message_lengths(grouplist[0], grouplist[1], data))
        print()
        counter += 1


# Funktio piirtää histogrammin viestien kirjoitusajoista
def message_time_plot(data, grouplist=None):
    times = data['secsfrommidnight'].to_numpy()
    times_moded = data['secsfrom8am'].to_numpy()
    if not grouplist:
        fig, axes = matplotlib.pyplot.subplots()
        axes.hist(times, bins=10)
        matplotlib.pyplot.title("Viestien kirjoitusajat keskiyöstä")
        matplotlib.pyplot.ylabel('Lukumäärä')
        matplotlib.pyplot.xlabel('Viestin kirjoitusaika sekunteina')
        fig.savefig("msg_time_hist.png")
        fig_mod, axes_mod = matplotlib.pyplot.subplots()
        axes_mod.hist(times_moded, bins=10)
        matplotlib.pyplot.title("Viestien kirjoitusajat aamu kahdeksasta")
        matplotlib.pyplot.ylabel('Lukumäärä')
        matplotlib.pyplot.xlabel('Viestin kirjoitusaika sekunteina')
        fig_mod.savefig("msg_time_hist_mod.png")


# Funtkio laskee tunnuslukuja viestien kirjoitusajoista
def message_time_key_values(data):
    times = data['secsfrom8am'].to_numpy()
    mean = times.mean()
    print("Mean: " + str(mean))
    median = statistics.median(times)
    print("Median: " + str(median))
    sd = statistics.stdev(times)
    print("Standard deviation: " + str(sd))


# Funktio vertailee viestien kirjoitusaikoja annettujen kahden ryhmän välillä
def message_time_comparison(group1, group2, data):
    if group1 in GROUP_IDS and group2 in GROUP_IDS:
        x1 = data.loc[data['groupID'] == GROUP_IDS[group1]]['secsfrom8am'].to_numpy()
        x2 = data.loc[data['groupID'] == GROUP_IDS[group2]]['secsfrom8am'].to_numpy()
    else:
        print("Group(s) is not in the data!")
        return
    result, pvalue = scipy.stats.ttest_ind(x1, x2)
    print("Mean for group \"" + group1 + "\": " + str(x1.mean()))
    print("Mean for group \"" + group2 + "\": " + str(x2.mean()))
    fig, axes = matplotlib.pyplot.subplots()
    axes.hist((x1, x2), histtype='bar')
    matplotlib.pyplot.legend([group1, group2])
    matplotlib.pyplot.title("Viestien kirjoitusajat ryhmissä \"" + group1 + "\" ja \"" + group2 + "\"")
    matplotlib.pyplot.ylabel('Lukumäärä')
    matplotlib.pyplot.xlabel('Viestin kirjoitusaika sekunteina')
    fig.savefig("msg_time_hist" + group1 + "_" + group2 + ".png")
    return result, pvalue


# Toimenpiteet harjoitustyön osaan 3, Kirjoitusajat
def part3(data):
    message_time_plot(data)
    message_time_key_values(data)
    print()
    print("T-test \"" + GROUPS3[0] + "\" and \"" + GROUPS3[1] + "\":")
    print(message_time_comparison(GROUPS3[0], GROUPS3[1], data))


# Funktio laskee korrelaation syötetyille sanoille. Funktiolle voi antaa myös uutisryhmän, jossa tarkastelu tehdään.
def word_count_correlation(word1, word2, data, group=None):
    if not group:
        if word1 in data.columns and word2 in data.columns:
            x1 = data[word1].to_numpy()
            x2 = data[word2].to_numpy()
            return numpy.corrcoef(x1, x2)
        else:
            print("Word(s) is not in the data!")
            return
    elif group in GROUP_IDS:
        if word1 in data.columns and word2 in data.columns:
            x1 = data.loc[data['groupID'] == GROUP_IDS[group]][word1].to_numpy()
            x2 = data.loc[data['groupID'] == GROUP_IDS[group]][word2].to_numpy()
            return numpy.corrcoef(x1, x2)
        else:
            print("Word(s) is not in the data!")
            return
    else:
        print("Group is not in the data!")
        return


# Toimenpiteet harjoitustyön osaan 4, Korrelaatiot
def part4(data):
    for words in WORDS4:
        print("Correlation between words \"" + words[0] + "\" and \"" + words[1] + "\":")
        print(word_count_correlation(words[0], words[1], data))
        print()
    print("Correlation between words \"" + WORDS4[0][0] + "\" and \"" + WORDS4[0][1] + "\" in the group "
                                                                                       "\"" + GROUPS4[0] + "\".")
    print(word_count_correlation(WORDS4[0][0], WORDS4[0][1], data, GROUPS4[0]))
    return


# Funktio suorittaa normaalisuuden testin sentimenttiarvolle ja piirtää histogrammin
def sentiment_normal_test(data):
    x = data['meanvalences'].to_numpy()
    result, pvalue = scipy.stats.normaltest(x, axis=0)
    fig, axes = matplotlib.pyplot.subplots()
    axes.hist(x)
    matplotlib.pyplot.title("Viestien sentimenttiarvot")
    matplotlib.pyplot.ylabel('Lukumäärä')
    matplotlib.pyplot.xlabel('Sentimenttiarvot')
    fig.savefig("sentiment_hist_.png")
    return [result, pvalue]


# Funktio etsii kolme suurimman ja pienimmän sentimenttiarvon omaavaa ryhmää
def find_min_max_sentiment(data):
    means = []
    groups = {}
    for group in GROUP_IDS:
        group_data_array = data.loc[data['groupID'] == GROUP_IDS[group]]['meanvalences'].to_numpy()
        mean = group_data_array.mean()
        means.append(mean)
        groups[mean] = group
    means.sort()
    max = {}
    min = {}
    for i in range(0, 3):
        min[means[i]] = groups[means[i]]
        max[means[len(means)-(i+1)]] = groups[means[len(means)-(i+1)]]
    return min, max


# Funktio laskee tunnuslukuja sentimenttiarvoille
def sentiment_key_values(data):
    key_value_dataframe = pandas.DataFrame(columns=['group', 'mean', 'median', 'SD', '25% quantile', '75% quantile'])
    for group in GROUP_IDS:
            group_data_array = data.loc[data['groupID'] == GROUP_IDS[group]]['meanvalences'].to_numpy()
            mean = group_data_array.mean()
            median = statistics.median(group_data_array)
            sd = statistics.stdev(group_data_array)
            quantiles = numpy.quantile(group_data_array, [0.25, 0.75])
            new_dataframe = pandas.DataFrame([[group, mean, median, sd, quantiles[0], quantiles[1]]],
                                             columns=['group', 'mean', 'median', 'SD', '25% quantile', '75% quantile'])
            key_value_dataframe = key_value_dataframe.append(new_dataframe, ignore_index=True)
    key_value_dataframe.to_csv("sentiment_values.csv", sep='&')
    return key_value_dataframe


# Funktio piirtää histogrammin sentimenttiarvoista annetussa uutisryhmässä
def sentiment_plot(group, data):
    values = []
    if group in GROUP_IDS:
        group_data_array = data.loc[data['groupID'] == GROUP_IDS[group]]['meanvalences'].to_numpy()
        values.append(group_data_array)
        fig, axes = matplotlib.pyplot.subplots()
        axes.hist(values, bins=10)
        matplotlib.pyplot.title("Sentimenttiarvo ryhmässä \"" + group + "\"")
        matplotlib.pyplot.ylabel('Lukumäärä')
        matplotlib.pyplot.xlabel('Sentimenttiarvo')
        fig.savefig("sentiment_hist_" + group + ".png")
    else:
        print("Group is not in the data!")


# Funktio vertailee sentimenttiarvojen jakaumia kahdessa annetussa uutisryhmässä
def sentiment_comparison(group1, group2, data):
    if group1 in GROUP_IDS and group2 in GROUP_IDS:
        x1 = data.loc[data['groupID'] == GROUP_IDS[group1]]['meanvalences'].to_numpy()
        x2 = data.loc[data['groupID'] == GROUP_IDS[group2]]['meanvalences'].to_numpy()
    else:
        print("Group(s) is not in the data!")
        return
    result, pvalue = scipy.stats.ttest_ind(x1, x2)
    print("Mean for group \"" + group1 + "\": " + str(x1.mean()))
    print("Mean for group \"" + group2 + "\": " + str(x2.mean()))
    return [result, pvalue]


# Toimenpiteet harjoitustyön osaan 5, Sentimentin analyysi
def part5(data):
    print("Normaltest for data:")
    print(sentiment_normal_test(data))
    negative, positive = find_min_max_sentiment(data)
    print("Three most positive news groups are:")
    counter = 1
    for value in positive:
        print(str(counter) + ". " + positive[value] + ", " + str(value))
        counter += 1
    counter = 1
    print()
    print("Three most negative news groups are:")
    for value in negative:
        print(str(counter) + ". " + negative[value] + ", " + str(value))
        counter += 1
    print()
    print(sentiment_key_values(data))
    print()
    for group in GROUP_IDS:
        sentiment_plot(group, data)
    for groups in GROUPS5:
        print("Test groups \"" + groups[0] + "\" and \"" + groups[1] + "\":")
        print(sentiment_comparison(groups[0], groups[1], data))
        print()


# Funktio poimii ainestosta halutun osan ja lisää niihin apumuuttujan ennustuksia varten
def initialize_data_for_part6(group1, group2, data):
    if group1 in GROUP_IDS and group2 in GROUP_IDS:
        new_dataframe = data.loc[(data['groupID'] == GROUP_IDS[group1]) | (data['groupID'] == GROUP_IDS[group2])]
        target = []
        for id in new_dataframe['groupID']:
            if id == GROUP_IDS[group1]:
                target.append(1)
            elif id == GROUP_IDS[group2]:
                target.append(-1)
        new_dataframe['target'] = target
        print(new_dataframe)
        print("Data initialized for procedures \n")
        return new_dataframe
    else:
        print("Group(s) is not in the data!")
        return []


# Funktio pyrkii ennustaa lineaarisella regressiolla tavoitemuuttujan arvoa
def group_prediction(words, group1, group2, data):
    x = []
    y = []
    for word in words:
        if word in data.columns:
            new_x = data.loc[(data['groupID'] == GROUP_IDS[group1]) | (data['groupID'] == GROUP_IDS[group2])][word].\
                to_numpy()
            new_y = data.loc[(data['groupID'] == GROUP_IDS[group1]) | (data['groupID'] == GROUP_IDS[group2])]['target']\
                .to_numpy()
            if len(x) < 1:
                x = new_x
                y = new_y
            else:
                x = x + new_x
        else:
            print("Word is not in the data!")
            return
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    fig, axes = matplotlib.pyplot.subplots()
    axes.scatter(x, y, s=20, c='r')
    result = sklearn.linear_model.LinearRegression().fit(x, y)
    y_pred = result.predict(x)
    mse = sklearn.metrics.mean_squared_error(y, y_pred)
    # Muutetaan ennustetut arvot binaarimuutujan arvoiksi, kuten kohdemuuttujakin on
    for i in range(len(y_pred)):
        if y_pred[i] < 0:
            y_pred[i] = -1
        elif y_pred[i] > 0:
            y_pred[i] = 1
    axes.scatter(x, y_pred, s=10, c='b')
    matplotlib.pyplot.legend(['Toteutuneet arvot', 'Ennustetut arvot'])
    matplotlib.pyplot.ylabel('Tavoitemuuttujan arvo')
    matplotlib.pyplot.xlabel('Sanan esiintymismäärä')
    matplotlib.pyplot.title('Ennustus, ' + str(words))
    fig.savefig('prediction_' + str(words) + '.png')
    return mse


# Toimenpiteet harjoitustyön osaan 6, Uutisryhmän ennustaminen
def part6(data):
    test_data = initialize_data_for_part6(GROUPS6[0], GROUPS6[1], data)
    for words in WORDS6:
        mse = group_prediction(words, GROUPS6[0], GROUPS6[1], test_data)
        print('Mse for prediction in ' + str(words) + ': ' + str(mse))
        print()


def main():
    data = read_file()
    if data[1]:
        print("Procedures for part 1 \n")
        part1(data[0])
        print("\n")
        print("Procedures for part 2 \n")
        part2(data[0])
        print("\n")
        print("Procedures for part 3 \n")
        part3(data[0])
        print("\n")
        print("Procedures for part 4 \n")
        part4(data[0])
        print("\n")
        print("Procedures for part 5 \n")
        part5(data[0])
        print("\n")
        print("Procedures for part 6 \n")
        part6(data[0])
    else:
        print("file not found!")


main()
