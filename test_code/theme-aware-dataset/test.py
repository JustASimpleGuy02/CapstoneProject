# %%
sub2cates = {
    'Ankle boots': 'footwear',
    'Chain bag': 'accessory',
    'Chiffon shirt': 'top',
    'Doudou shoes': 'footwear',
    'Flip-flops man': 'footwear',
    'High heel': 'footwear',
    'Martin boots': 'footwear',
    "Men's T-shirt": 'top',
    "Men's handbags": 'bag',
    "Men's shirts": 'top',
    "Men's slippers": 'footwear',
    "Men's vest": 'top',
    'Roman shoes': 'footwear',
    'Rompers': 'full-body',
    'Shoulder Bags': 'bag',
    'Shoulder Messenger Bag': 'bag',
    'Small square bag': 'bag',
    'Sweatpants': 'bottom',
    'Tote': 'bag',
    'Wedges': 'footwear',
    'Wide leg pants': 'bottom',
    "Women's T-shirt": 'top',
    'accessories': 'accessory',
    'backpack': 'bag',
    'blouses': 'top',
    'boot cut pant': 'bottom',
    'briefcase': 'bag',
    'bucket bag': 'bag',
    'canvas bag': 'bag',
    'canvas shoes': 'footwear',
    'casual pants': 'bottom',
    'casual shoes': 'footwear',
    'chelsea boots': 'footwear',
    'clutch bag': 'bag',
    'coat for male': 'outerwear',
    'crossbody bag': 'bag',
    'dress shoes': 'footwear',
    'fish mouth shoes': 'footwear',
    'flip flops': 'footwear',
    'handbag': 'bag',
    'jeans': 'bottom',
    'laptop bag': 'bag',
    'loafers': 'footwear',
    'man shoes': 'footwear',
    "men's backpack": 'bag',
    "men's business casual shoes": 'footwear',
    "men's canvas shoes": 'footwear',
    "men's cardigan": 'top',
    "men's casual pants": 'bottom',
    "men's casual shoes": 'footwear',
    "men's coat": 'outerwear',
    "men's down jacket": 'outerwear',
    "men's dress pants": 'bottom',
    "men's jacket": 'outerwear',
    "men's jeans": 'bottom',
    "men's polo shirt": 'top',
    "men's sandals": 'footwear',
    "men's sneakers": 'footwear',
    "men's sports bag": 'bag',
    "men's sports shoes": 'footwear',
    "men's suit": 'full-body',
    "men's suits": 'full-body',
    "men's sweater": 'top',
    "men's sweatpants": 'bottom',
    "men's sweatshirt": 'top',
    "men's trench coat": 'outerwear',
    "men's vest": 'top',
    "men's wallet": 'accessory',
    'messenger bag': 'bag',
    'mules': 'footwear',
    'platform shoes': 'footwear',
    'sandals': 'footwear',
    'shell bag': 'bag',
    'single shoes': 'footwear',
    'snow boots': 'footwear',
    'wallet': 'accessory',
    "women's boots": 'footwear',
    "women's bottoming shirt": 'top',
    "women's cardigan": 'top',
    "women's casual pants": 'bottom',
    "women's cloak": 'outerwear',
    "women's coat": 'outerwear',
    "women's down jacket": 'outerwear',
    "women's dress": 'full-body',
    "women's jacket": 'outerwear',
    "women's overalls": 'full-body',
    "women's skirt": 'bottom',
    "women's suit": 'full-body',
    "women's suits": 'full-body',
    "women's sweater": 'top',
    "women's sweatshirt": 'top',
    "women's trench coat": 'outerwear',
    "women's vest": 'top',
    'work shoes': 'footwear'
}

# %%
cate2subs = {}
for k, v in sub2cates.items():
    if v not in cate2subs:
        cate2subs[v] = []
    cate2subs[v].append(k)
cate2subs

# %%
