import cv2 as cv
from time import time, gmtime, strftime
from random import choice,randint, choices, seed
import math
import numpy as np
from perlin_numpy import generate_fractal_noise_2d

PATH = '/your-end-directory'

def create_tile(r,g,b):
    image = np.zeros((2, 2, 3), np.uint8)
    color = tuple(reversed((r,g,b)))
    image[:] = color
    return image

sizes = {
    '2048x2048':{"res":(1024,1024),"crop":False},
    '1920x1080':{"res":(768,1024),"crop":[540,960]},
    '1280x720':{"res":(512,768),"crop":[360,640]},
    '1024x1024':{"res":(512,512),"crop":False},
    '640x480':{"res":(256,512),"crop":[240,320]},
    '600x600':{"res":(512,512),"crop":[300,300]},
}

map_types = {
    'normal':{
        'ocean_layers':['sand1','sea1','sea4'],
        'land_levels':[1,0.1,-1.3],
        'land_layers':['snow1','rock1','land1','sea1'],
        'pass1':{'thresholds':[5,8],'weights':(2,5,1)},
        'pass2':{'weights':(1,5,3),'result':0},
        'blend_layers':['rock','land'],
        'blend_weights':[(3,5,3),(5,2,3),(4,1,1)],
        'river_pass':{'thresholds':[2,10,30]},
        'city_pass':{'state':'land1','threshold':27,'frequency_modifier':10}
    },
    'desert':{ # needs work (less burnt out)
        'ocean_layers':['sand2','sand1','sand1'],
        'land_levels':[1,0.2,-1.5],
        'land_layers':['sand3','rock2','sand1','sea1'],
        'pass1':{'thresholds':[0,100],'weights':(5,4,8)},
        'pass2':{'weights':(1,5,10),'result':0},
        'blend_layers':['land','sand'],
        'blend_weights':[(3,4,3),(5,2,3),(4,1,2)],
        'river_pass':{'thresholds':[0,13,-1]},
        'city_pass':{'state':'rock2','threshold':0,'frequency_modifier':10}
    },
    'rocky':{ # needs work (less garbage)
        'ocean_layers':['rock1','sea1','sea4'],
        'land_levels':[1,-0.2,-1],
        'land_layers':['sand1','rock1','sand3','sea1'],
        'pass1':{'thresholds':[5,8],'weights':(2,5,1)},
        'pass2':{'weights':(7,1,4),'result':1},
        'blend_layers':['land','rock'],
        'blend_weights':[(3,5,3),(5,4,2),(4,3,1)],        
        'river_pass':{'thresholds':[2,10,15]},
        'city_pass':{'state':'sea3','threshold':0,'frequency_modifier':5}
    },
    'oceania':{
        'ocean_layers':['sand1','sea1','sea4'],
        'land_levels':[1.3,0.4,0.2],
        'land_layers':['rock1','land1','sand1','sea1'],
        'pass1':{'thresholds':[10,30],'weights':(15,9,1)},
        'pass2':{'weights':(0,1,1),'result':1},
        'blend_layers':['sea','land'],
        'blend_weights':[(0,5,3),(0,3,2),(0,1,1)],
        'river_pass':{'thresholds':[2,8,30]},
        'city_pass':{'state':'sea3','threshold':20,'frequency_modifier':0}
    },
    'jungle':{
        'ocean_layers':['sand1','sea1','sea4'],
        'land_levels':[1.3,0.6,-0.5],
        'land_layers':['snow1','rock1','land1','sea1'],
        'pass1':{'thresholds':[10,30],'weights':(9,5,1)},
        'pass2':{'weights':(0,0,1),'result':1},
        'blend_layers':['land','rock'],
        'blend_weights':[(0,5,3),(0,3,2),(0,1,1)],
        'river_pass':{'thresholds':[2,8,30]},
        'city_pass':{'state':'sea3','threshold':20,'frequency_modifier':0}
    },
    'ice':{
        'ocean_layers':['snow1','sea3','sea1'],
        'land_levels':[1,-0.2,-1.3],
        'land_layers':['snow2','snow1','rock1','sea3'],
        'pass1':{'thresholds':[0,100],'weights':(0,10,1)},
        'pass2':{'weights':(1,5,10),'result':1},
        'blend_layers':['land','rock'],
        'blend_weights':[(1,10,3),(1,5,2),(1,3,1)],
        'river_pass':{'thresholds':[2,10,20]},
        'city_pass':{'state':'rock1','threshold':2,'frequency_modifier':10}
    },
    # 'helper_template':{
    #     'ocean_layers':['top','middle','bottom'],
    #     'land_levels':[top,middle,bottom],
    #     'land_layers':['top','mid1','mid2','bottom'],
    #     'pass1':{'thresholds':[sea1 between],'weights':(cell,sea3,sand2)},
    #     'pass2':{'weights':(sea2,rock2,land2),'result':circle,line},
    #     'blend_layers':[cell,x3+4+1,y3+4+1],
    #     'river_pass':{'thresholds':[sea3 between,land1 count]},
    #     'city_pass':{'state':cell,'threshold':sea1 count,'frequency_modifier':rand}
    # },
}


images = {
            'sea1':create_tile(110, 170, 255),
            'sea2':create_tile(130, 190, 255),
            'sea3':create_tile(140, 200, 255),
            'sea4':create_tile(90, 150, 255),
            'sand1':create_tile(250, 240, 180),
            'sand2':create_tile(230, 210, 160),
            'sand3':create_tile(210, 190, 120),
            'sand4':create_tile(200, 180, 110),
            'land1':create_tile(64, 128, 62),
            'land2':create_tile(47, 105, 46),
            'land3':create_tile(38, 92, 37),
            'land4':create_tile(32, 82, 31),
            'rock1':create_tile(89, 84, 76),
            'rock2':create_tile(87, 79, 67),
            'rock3':create_tile(82, 72, 57),
            'rock4':create_tile(77, 65, 48),
            'snow1':create_tile(230, 230, 230),
            'snow2':create_tile(220, 220, 220),
            'snow3':create_tile(210, 210, 210),
            'city':create_tile(60,60,60),
            'black':create_tile(100, 100, 100),
            'white':create_tile(130, 190, 200),
            }


# Returns an array corresponding to a square of {radius} around y,x
def scan_square(array, col_index, row_index, radius):
    result = []
    for y in range(row_index - radius, row_index + radius):
        for x in range(col_index - radius, col_index + radius):
            try:
                result.append(array[y][x])
            except:
                continue
    return result


# Returns the array, with a square of {radius} around y,x painted in {paint}
def paint_square(array, col_index, row_index, radius, paint):
    for y in range(row_index - radius, row_index + radius):
        for x in range(col_index - radius, col_index + radius):
            try:
                array[y][x] = paint
            except:
                continue
    return array


# Returns an array corresponding to a shape around y,x
def scan_lines(array, shapes, col_index, row_index, radius):
    coords = []
    result = []
    shapes = ['|','-'] if shapes == '+' else shapes
    shapes = ['\\','/'] if shapes == 'x' else shapes
    shapes = ['|','-','\\','/'] if shapes == '*' else shapes
    for i in range(-radius,radius):
        if '|' in shapes:
            coords.extend([(-i,0),(i,0)])
        if '-' in shapes:
            coords.extend([(0,-i),(0,i)])
        if '\\' in shapes:
            coords.extend([(-i,-i),(i,i)])
        if '/' in shapes:
            coords.extend([(-i,i),(i,-i)])
        for (y,x) in coords:
            try:
                result.append(array[row_index+y][col_index+x])
            except:
                continue

    return result


# Returns the array, with a shape painted in {paint} around y,x
def paint_lines(array, shapes, col_index, row_index, radius, paint):
    coords = []
    shapes = ['|','-'] if shapes == '+' else shapes
    shapes = ['\\','/'] if shapes == 'x' else shapes
    shapes = ['|','-','\\','/'] if shapes == '*' else shapes
    for i in range(-radius,radius):
        if '|' in shapes:
            coords.extend([(-i,0),(i,0)])
        if '-' in shapes:
            coords.extend([(0,-i),(0,i)])
        if '\\' in shapes:
            coords.extend([(-i,-i),(i,i)])
        if '/' in shapes:
            coords.extend([(-i,i),(i,-i)])
        for (y,x) in coords:
            try:
                array[row_index+y][col_index+x] = paint
            except:
                continue
    return array


# Returns an array corresponding to a circle of {radius} around y,x
def scan_circle(array, col_index, row_index, radius, paint):
    result = []
    for y in range(row_index - radius, row_index + radius):
        for x in range(col_index - radius, col_index + radius):
            if math.sqrt((row_index - y) ** 2 + (col_index - x) ** 2) <= radius:
                try:
                    result.append(array[y][x])
                except:
                    continue
    return result


# Returns the array, with a circle of {radius} around y,x painted in {paint}
def paint_circle(array, col_index, row_index, radius, paint):
    for y in range(row_index - radius, row_index + radius):
        for x in range(col_index - radius, col_index + radius):
            if math.sqrt((row_index - y) ** 2 + (col_index - x) ** 2) <= radius:
                try:
                    array[y][x] = paint
                except:
                    continue
    return array


# Creates the map image
def create_map(template='normal',size='1024x1024',cities=True,map_seed=None):
    map_type= map_types[template]
    map_seed = map_seed if map_seed else randint(1,10000)

    # Set seed
    seed(map_seed)
    np.random.seed(map_seed)

    # Ocean map (background)
    ocean_noise = generate_fractal_noise_2d (sizes[size]['res'], (8, 8),6,0.9,tileable=(0,0))
    ocean_map = ocean_noise.tolist()
    for y,row in enumerate(ocean_map):
        for x,cell in enumerate(row):
            if cell > 1.4:
                ocean_map[y][x] = map_type['ocean_layers'][0]
            elif -0.3 < cell < 1.4:
                ocean_map[y][x] = map_type['ocean_layers'][1]
            else:
                ocean_map[y][x] = map_type['ocean_layers'][2]

    # Land map (foreground)
    land_noise = generate_fractal_noise_2d (sizes[size]['res'], (8, 8),6,0.9,tileable=(0,0))
    land_map = land_noise.tolist()
          
    for y,row in enumerate(land_map):
        for x,cell in enumerate(row):
            if cell > map_type['land_levels'][0]:
                land_map[y][x] = map_type['land_layers'][0]
            elif map_type['land_levels'][1] < cell < map_type['land_levels'][0]:
                land_map[y][x] = map_type['land_layers'][1]
            elif map_type['land_levels'][2] < cell < map_type['land_levels'][1]:
                land_map[y][x] = map_type['land_layers'][2]
            else:
                land_map[y][x] = map_type['land_layers'][3]

    # Ocean + Land merge
    
    merge_noise = generate_fractal_noise_2d (sizes[size]['res'], (2, 2),4,0.5,tileable=(0,0))
    merge = merge_noise.tolist()
    
    # Crop
    if sizes[size]['crop']:
        ocean_map = ocean_map[0:sizes[size]['crop'][0]]
        for index,row in enumerate(ocean_map):
            ocean_map[index] = row[0:sizes[size]['crop'][1]]
        land_map = land_map[0:sizes[size]['crop'][0]][0:sizes[size]['crop'][1]]
        for index,row in enumerate(land_map):
            land_map[index] = row[0:sizes[size]['crop'][1]]
        merge = merge[0:sizes[size]['crop'][0]][0:sizes[size]['crop'][1]]
        for index,row in enumerate(merge):
            merge[index] = row[0:sizes[size]['crop'][1]]


    land = [] 
    for y,row in enumerate(merge):
        for x,cell in enumerate(row):
            if cell >0:
                merge[y][x] = land_map[y][x]
                land.append((x,y))
            else:
                merge[y][x] = ocean_map[y][x]

    # Optional - store rough map as map.png
    # merge_img = cv.vconcat([cv.hconcat([images[image] for image in row]) for row in merge])
    # cv.imwrite(PATH+'map.png', merge_img)

    # Beautify
    for (x,y) in land:
        cell = merge[y][x]
        surroundings = scan_square(merge,x,y,3)

        # Pass1
        if map_type['pass1']['thresholds'][0] < surroundings.count('sea1') < map_type['pass1']['thresholds'][1]:
            merge[y][x] = choices([cell,'sea3','sand2'],weights=map_type['pass1']['weights'],k=1)[0]
            
        # Pass2
        elif cell not in ['sea1','sea2','sand1','sand2']:
            if 'sea2' in surroundings or 'sand2' in surroundings:
                merge[y][x] = choices(['sea2','rock2','land2'],weights=map_type['pass2']['weights'],k=1)[0]
                if randint(0,100) == 0:
                    merge = [
                        paint_circle(merge,x,y,randint(2,3),'sea1'),
                        paint_lines(merge,choice(['|','-','\\','/']),x,y,randint(1,2),'sea3')
                        ][map_type['pass2']['result']]
                    
        #     # Blending
            if cell not in ['rock2','land2'] and (
                'rock2' in surroundings or 'land2' in surroundings):
                merge[y][x] = choices(
                    [cell,map_type['blend_layers'][0]+'3',map_type['blend_layers'][1]+'3'],
                    weights=map_type['blend_weights'][0],k=1)[0]
            if cell not in ['rock2','land2','rock3','land3'] and (
                'rock3' in surroundings or 'land3' in surroundings):
                merge[y][x] = choices(
                    [cell,map_type['blend_layers'][0]+'4',map_type['blend_layers'][1]+'4'],
                    weights=map_type['blend_weights'][1],k=1)[0]
            if cell not in ['rock2','land2','rock3','land3','rock4','land4'] and (
                'rock4' in surroundings or 'land4' in surroundings):
                merge[y][x] = choices(
                    [cell,map_type['blend_layers'][0]+'1',map_type['blend_layers'][1]+'1'],
                    weights=map_type['blend_weights'][2],k=1)[0]

        #     # Rivers
            surroundings = scan_square(merge,x,y,4)
            if map_type['river_pass']['thresholds'][0] < surroundings.count('sea3') < map_type['river_pass']['thresholds'][1] and surroundings.count('land1') > map_type['river_pass']['thresholds'][2]:
                if randint(0,10) == 10:
                    merge = choices([
                        paint_square(merge,x,y,1,'sea2'),
                        paint_lines(merge,choice(['|','-','\\','/']),x,y,randint(1,2),'sea3')
                        ],weights=(1,3),k=1)[0]

        #     # City Pass
            if cities:
                if cell == map_type['city_pass']['state'] and surroundings.count('sea1') > map_type['city_pass']['threshold']:
                    if randint(0,map_type['city_pass']['frequency_modifier']) == 0:
                        merge = paint_square(merge,x,y,randint(2,3),'city') if 10 < surroundings.count('city') < 30 else choices([paint_square(merge,x,y,randint(1,2),'city'),paint_lines(merge,choice(['|','-','\\','/']),x,y,randint(1,2),'city')],weights=(1,3),k=1)[0]


    # Store coastal pass image
    map_img = cv.vconcat([cv.hconcat([images[image] for image in row]) for row in merge])
    filename = 'map-'+str(int(time()))+'-'+str(map_seed)+'.png'
    cv.imwrite(PATH+filename, map_img)
    

create_map()
