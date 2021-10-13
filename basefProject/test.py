#Find resources in the user's vicinity using Yelp API
import requests 

location = "Brantford, Ontario"
category = "c_and_mh"
term = "mental health"
url = 'https://api.yelp.com/v3/businesses/search'
key = open(r'/Users/gurnirmalkaur/Desktop/key.txt').readlines()[0]

def find_resource(location):
    resources = ""
    headers = {
    'Authorization': 'Bearer %s' % key
    }       
    parameters = {'location': location,
                'categories': category,
                'term': term,
                'limit': 3}
    response = requests.get(url, headers=headers, params=parameters)
    resource = response.json()['businesses']

    for r in resource:
        name = r['name']
        location = r['location']['address1']
        city = r['location']['city']
        province = r['location']['state']
        resource = (f'{name}, {location}, {city}, {province}\n')
        resources += resource 
    return resources

