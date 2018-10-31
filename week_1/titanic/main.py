import pandas
import re
import pdb

data = pandas.read_csv(
    '/Users/markfrost/learn/course_machine_learning/week_1/titanic/data/train.csv', index_col='PassengerId')

print(data)

print('================== task 1')


male = data['Sex'].value_counts()['male']
female = data['Sex'].value_counts()['female']
print("male: %(male)s female: %(female)s" % locals())

print('================== task 2')

survived_percent = round(data['Survived'].value_counts(
)[1] / data['Survived'].count() * 100, 2)

print("survived: %(survived_percent)s " % locals())


print('================= task 3')

percent_first_class = round(data['Pclass'].value_counts()[
                            1] / data['Pclass'].count() * 100, 2)

print("first class percent: %(percent_first_class)s " % locals())

print('================= task 4')

average_age = round(data['Age'].sum() / data['Age'].count(), 2)
median_age = data['Age'].median()

print("average Age: %(average_age)s median Age: %(median_age)s" % locals())

print('================= task 5')

corr_pearson = round(data['Age'].sum() / data['Age'].count(), 2)

# print("average Age: %(average_age)s median Age: %(median_age)s" % locals())


print('================= task 6')


def sanitize(full_name):
    matched = re.search(r'\((\w*)', full_name)
    name = ''
    if re.search(r'\((\w*)', full_name) is not None:
        name = matched[1]
    else:
        name = full_name.split(' ')[2].replace('(', '').replace(')', '')
    return name


data = data.loc[
    data['Sex'] == 'female']


data['Name'] = data['Name'].apply(sanitize)
print(data.agg(lambda x: x.value_counts().index[0]))
