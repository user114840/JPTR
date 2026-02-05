
import math
# sampler for batch generation

def haversine(lat1, lon1, lat2, lon2):
    # Convert lat/lon to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Compute deltas
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine distance
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = 6371 * c  # earth radius in km
    return distance
def find_nearest_pois(rated_pois, ground_truth, all_pois):
    # Compute distances between unseen POIs and the ground-truth POI
    distances = [(poi_id, haversine(all_pois[ground_truth]['latitude'], all_pois[ground_truth]['longitude'],
                                     all_pois[poi_id]['latitude'], all_pois[poi_id]['longitude']))
                 for poi_id in all_pois if poi_id not in rated_pois and poi_id != ground_truth]

    # Sort by distance and pick the closest 100
    nearest_pois = sorted(distances, key=lambda x: x[1])[:100]
    return [poi[0] for poi in nearest_pois]

def find_nearest_negative_poi(ground_truth, all_pois):
    # Compute distances between all other POIs and the ground-truth POI
    distances = [(poi_id, haversine(all_pois[ground_truth]['latitude'], all_pois[ground_truth]['longitude'],
                                     all_pois[poi_id]['latitude'], all_pois[poi_id]['longitude']))
                 for poi_id in all_pois if poi_id != ground_truth]

    # Sort by distance and pick the closest 1000
    #print(len(distances))
    nearest_pois = sorted(distances, key=lambda x: x[1])[:1000]
    return [poi[0] for poi in nearest_pois]
