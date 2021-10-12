#Find resources in the user's vicinity using Yelp API

import requests 
location = "Brantford, Ontario"
term = "mall"
url = 'https://api.yelp.com/v3/businesses/search'
key = open(r'/Users/gurnirmalkaur/Desktop/key.txt').readlines()[0]
headers = {
    'Authorization': 'Bearer %s' % key
}
parameters = {'location': location,
              'term': term,
              'radius': 10000,
              'limit': 10}

response = requests.get(url, headers=headers, params=parameters)
resource = response.json()['businesses']

for r in resource:
    name = r['name']
    location = r['location']['address1']
    print(f'{name}, {location}')
