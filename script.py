from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = fetch_20newsgroups()
print(emails.target_names)

# compares distinguishes between hockey and pc hardware emails with 99% accuracy

# set train emails from subset hockey and pc hardware
train_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'],
                                  subset='train',
                                  shuffle=True,
                                  random_state=108
                                  )
# set test emails from subset hockey and pc hardware
test_emails = fetch_20newsgroups(categories=['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'],
                                  subset='test',
                                  shuffle=True,
                                  random_state=108
                                  )
# inspecting data structure
print(train_emails.data[5])
print(train_emails.target[5])
print(train_emails.target_names)

# count all words in emails
counter = CountVectorizer()
counter.fit(test_emails.data, train_emails.data)

train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)

# print accuracy of distinguishing between two email types
print(classifier.score(test_counts, test_emails.target))
