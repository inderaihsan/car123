import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import requests

# Set page configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("Car Price Prediction")
st.markdown("Enter the details of the car to predict its price")

# Load the model


# Define location categories based on the model
locations = ['Jawa Barat', 'DKI Jakarta', 'Banten', 'Jawa Tengah', 'Jawa Timur', 
             'Sumatera Selatan', 'Yogyakarta', 'Kalimantan Barat', 'Sumatera Utara', 
             'Bali', 'Sulawesi Selatan', 'Jambi', 'Bangka Belitung', 'Riau', 
             'Nangroe Aceh Darussalam', 'Kalimantan Timur', 'Kalimantan Tengah', 
             'Kepulauan Riau', 'Sulawesi Tengah', 'Kalimantan Selatan', 'Sumatera Barat', 
             'Sulawesi Utara', 'Gorontalo', 'Lampung']

# Define brand categories based on the model
japan_brands = ['Toyota', 'Honda', 'Nissan', 'Suzuki', 'Mitsubishi', 'Mazda', 'Subaru', 'Daihatsu', 'Isuzu', 'Infiniti', 'Datsun']
korea_brands = ['Hyundai', 'KIA', 'Genesis']
germany_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Volkswagen', 'Porsche', 'smart']
usa_brands = ['Ford', 'Chevrolet', 'Dodge', 'RAM', 'Tesla', 'Jeep', 'Hummer', 'GMC', 'Cadillac']
uk_brands = ['MINI', 'Rolls-Royce', 'Jaguar', 'Bentley', 'Aston', 'Morgan', 'McLaren', 'Land']
china_brands = ['Wuling', 'DFSK', 'BYD', 'Geely', 'GWM', 'Denza', 'GAC']
france_brands = ['Peugeot', 'Renault', 'Citroen']
italy_brands = ['Fiat', 'Alfa', 'Ferrari', 'Maserati', 'Abarth']
other_brands = ['Proton', 'Vinfast', 'Volvo', 'Ineos']
luxury_brands = ['Porsche', 'Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Jaguar', 'Land', 'Rolls-Royce', 'Bentley', 'Ferrari',
                'Maserati', 'McLaren', 'Genesis', 'Cadillac', 'Tesla', 'Aston', 'MINI']
sport_car_brands = ['McLaren', 'Ferrari', 'Lamborghini', 'Porsche']

# Combine all brands
all_brands = list(set(japan_brands + korea_brands + germany_brands + usa_brands + uk_brands + 
                      china_brands + france_brands + italy_brands + other_brands))
brands = sorted(all_brands)

# Define car types for each brand (simplified for demo)
car_type_mapping = {
    'Honda': ['Accord', 'BR-V', 'Brio', 'CR-V', 'CR-Z', 'City', 'Civic', 'Elysion', 'Freed', 'HR-V', 'Jazz', 'Mobilio', 'Odyssey', 'Prelude', 'S 660', 'Stream', 'WR-V'],
    'Porsche': ['718', '911', 'Boxster', 'Cayenne', 'Cayman', 'Macan', 'Panamera', 'Taycan 4S Cross Turismo Coupe', 'Taycan 4S Performance Battery Plus Sedan', 'Taycan 4S Performance Battery Sedan', 'Taycan Performance Battery Sedan', 'Taycan Turbo Sedan'],
    'Nissan': ['370Z', 'Elgrand', 'Evalia', 'Frontier', 'Frontier Navara', 'Grand Livina', 'Juke', 'Kicks', 'Leaf (1 Tone) Hatchback', 'Livina', 'Livina X-Gear', 'Magnite', 'March', 'Murano', 'Navara', 'Patrol', 'Serena', 'Teana', 'Terra', 'Terrano', 'X-Trail', 'Z'],
    'Hyundai': ['', 'Avega', 'Creta', 'Excel', 'Getz', 'Grand Avega', 'Grand i10', 'H-1', 'H100', 'IONIQ 5 Batik Edition SUV', 'IONIQ 5 Long Range Signature SUV', 'IONIQ 5 Prime Long Range SUV', 'IONIQ 5 Prime Standard Range SUV', 'IONIQ 5 Signature Long Range SUV', 'IONIQ 5 Signature Standard Range SUV', 'IONIQ 6 Fastback', 'IONIQ 6 Signature Long Range Sedan', 'IONIQ Electric Signature Fastback', 'IONIQ Signature Coupe', 'Kona', 'Kona EV SUV', 'Kona N-Line SUV', 'Kona Prime Long Range SUV', 'Kona Prime Standard Range SUV', 'Kona Signature Long Range SUV', 'Kona Signature SUV', 'Kona Signature Standard Range SUV', 'Kona Signature Wagon', 'Palisade', 'Santa Fe', 'Stargazer', 'Stargazer X', 'Staria', 'Trajet', 'Tucson', 'Venue', 'i20'],
    'Toyota': ['86', 'Agya', 'Alphard', 'Avanza', 'BZ4X (1 Tone) SUV', 'BZ4X (2 Tone) SUV', 'BZ4X Wagon', 'C-HR', 'Calya', 'Camry', 'Camry Hybrid', 'Celica', 'Corolla', 'Corolla Altis', 'Corolla Cross', 'Crown', 'Dyna', 'Estima', 'Etios', 'Etios Valco', 'FJ Cruiser', 'Fortuner', 'GR 86', 'GR Corolla', 'GR Supra', 'GR Yaris', 'GranAce', 'Harrier', 'Hiace', 'Hilux', 'Hilux Rangga', 'IST', 'Innova Venturer', 'Kijang', 'Kijang Innova', 'Kijang Innova Zenix', 'Land Cruiser', 'Land Cruiser Cygnus', 'Land Cruiser Prado', 'Limo', 'Markx', 'NAV1', 'Noah', 'Previa', 'Prius', 'RAV4', 'Raize', 'Rush', 'Sienta', 'Soluna', 'Starlet', 'Vellfire', 'Veloz', 'Vios', 'Voxy', 'Yaris', 'Yaris Cross', 'iQ'],
    'Suzuki': ['APV', 'Amenity', 'Baleno', 'Carry', 'Carry Real', 'Ciaz', 'Ertiga', 'Escudo', 'Esteem', 'Every', 'Grand Vitara', 'Ignis', 'Jimny', 'Karimun', 'Karimun Wagon R', 'Katana', 'Mega Carry', 'S-Presso', 'SX4', 'SX4 S-Cross', 'Sidekick', 'Splash', 'Swift', 'XL7'],
    'Mitsubishi': ['Canter', 'Colt', 'Colt Diesel', 'Colt L300', 'Colt T120SS', 'Delica', 'Eclipse Cross', 'Fuso Canter', 'Galant', 'Grandis', 'Kuda', 'Lancer', 'Mirage', 'Outlander', 'Outlander Sport', 'Pajero', 'Pajero Sport', 'Strada Triton', 'Triton', 'XFORCE', 'Xpander', 'Xpander Cross'],
    'Mercedes-Benz': ['190E', '280GE', '300', '300CE', 'A 200', 'A 250', 'A35 AMG', 'A45 AMG', 'AMG CLA45', 'AMG CLS63', 'AMG GLA35', 'AMG GLA45', 'AMG GLE43', 'AMG GLE53', 'AMG GT', 'AMG SL43', 'AMG SLC43', 'B170', 'B200', 'C 180', 'C 200', 'C 230', 'C 240', 'C 250', 'C 280', 'C 300', 'C 350', 'C200K', 'C43 AMG', 'C63 AMG', 'CL500', 'CLA200', 'CLA45 AMG', 'CLE 300', 'CLK230K', 'CLS 350', 'CLS 400', 'CLS 500', 'E 200', 'E 220', 'E 230', 'E 240', 'E 250', 'E 280', 'E 300', 'E 320', 'E 350', 'E 400', 'E200K', 'E43 AMG', 'E53 AMG', 'E63 AMG', 'EQB250 Progressive Line SUV', 'EQE350+ Electric Art Line Sedan', 'EQS 450 Electric 4MATIC AMG Line Wagon', 'EQS450 AMG Line 4MATIC SUV', 'EQS450+ Edition One Sedan', 'EQS450+ Electric Art Line Sedan', 'G 300', 'G 350', 'G 400', 'G 450', 'G 500', 'G55 AMG', 'G580 EQ SUV', 'G63 AMG', 'GL400', 'GL500', 'GL63', 'GLA 200', 'GLA 45', 'GLB200', 'GLC 200', 'GLC 250', 'GLC 300', 'GLE 250', 'GLE 400', 'GLE 43', 'GLE 450', 'GLE 53', 'GLS 400', 'GLS 450', 'Gls400', 'ML250', 'ML320', 'ML350', 'ML400', 'ML63', 'Maybach S500', 'Maybach S560', 'Maybach S580', 'R280L', 'S 280', 'S 300L', 'S 320', 'S 350', 'S 400', 'S 400L', 'S 450L', 'S 600', 'S 600L', 'S300 L', 'S350 L', 'S400 L', 'S450 L', 'S500 L', 'SL300', 'SL350', 'SL400', 'SL500', 'SLC200', 'SLK200', 'SLK200K', 'SLK230', 'SLK250', 'SPRINTER', 'Sprinter', 'V220', 'V250', 'V260', 'Vito'],
    'BMW': ['1 Series M', '116i', '218i', '220i', '318i', '320d', '320i', '323i', '325i', '328i', '330i', '335i', '428i', '430i', '435i', '520d', '520i', '523i', '528i', '530i', '535i', '550i', '630i', '640i', '645Ci', '730Ld', '730Li', '735i', '740Li', '750Li', '760Li', '840i', 'M135i', 'M2', 'M235i', 'M3', 'M4', 'M5', 'M6', 'M8', 'M850i', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'XM', 'Z4', 'i4 eDrive35 Coupe', 'i4 eDrive35 Gran Coupe', 'i4 eDrive40 M Sport Coupe', 'i4 eDrive40 M Sport Gran Coupe', 'i5 eDrive40 M Sport Sedan', 'i7 xDrive60 Gran Lusso Sedan', 'i8', 'iX xDrive40 Sport SUV', 'iX xDrive40 Sport Wagon', 'iX1 eDrive20 M Sport SUV'],
    'Daihatsu': ['Ayla', 'Feroza', 'Gran Max', 'Luxio', 'Rocky', 'Sigra', 'Sirion', 'Taft', 'Tanto', 'Taruna', 'Terios', 'Xenia', 'Zebra', 'Zebra Espass'],
    'Mazda': ['2', '3', '6', '8', 'Biante', 'CX-3', 'CX-30', 'CX-5', 'CX-60', 'CX-7', 'CX-8', 'CX-80', 'CX-9', 'MX-5', 'RX-8'],
    'Audi': ['A 1', 'A 3', 'A 4', 'A 5', 'A 6', 'A 8', 'Q3', 'Q5', 'Q7', 'Q8', 'R8', 'RS4', 'RS5', 'RS6', 'TT'],
    'KIA': ['Carens', 'Carnival', 'EV6 GT-Line SUV', 'EV9 GT-Line SUV', 'Grand Carnival', 'Grand Sedona', 'Picanto', 'Pride', 'Rio', 'Seltos', 'Sonet', 'Sonet 7', 'Sorento', 'Sportage', 'Travello'],
    'Volkswagen': ['Caravelle', 'Golf', 'ID. Buzz Life Van', 'ID. Buzz Style Van', 'NBeetle', 'Polo', 'Scirocco', 'T Cross', 'T-Cross', 'TBeetle', 'Tiguan', 'Touareg', 'Touran'],
    'Chery': ['Omoda 5', 'Omoda E5 EV Pure SUV', 'Omoda E5 EV SUV', 'Omoda E5 EV Wagon', 'Omoda J6 EV IWD Phantom Edition SUV', 'Omoda J6 EV IWD SUV', 'Omoda J6 EV RWD Phantom Edition SUV', 'Omoda J6 EV RWD SUV', 'Tiggo 5x', 'Tiggo 7 Pro', 'Tiggo 8', 'Tiggo 8 Pro', 'Tiggo Cross'],
    'MINI': ['', 'Aceman SE Hatchback', 'Cabrio', 'Clubman', 'Cooper', 'Cooper Cooper SE Hatchback', 'Cooper S Electric Level 3 Hatchback', 'Cooper S Electric Resolute Edition 3 Door Hatchback', 'Countryman', 'Countryman SE ALL4 SUV', 'Paceman', 'Roadster'],
    'Lexus': ['CT 200h', 'ES 250', 'ES 300h', 'GS 200t', 'GS 300', 'GX 550', 'LM 350', 'LM 350h', 'LM350', 'LM350h', 'LS 460L', 'LS 500', 'LS 600hL', 'LX 570', 'LX 600', 'NX 200T', 'NX 200t', 'NX 300', 'NX 350h', 'RC F', 'RX 200t', 'RX 270', 'RX 300', 'RX 330', 'RX 350', 'RX 350h', 'RX 450h+', 'RX 500H', 'RZ 450E Luxury SUV', 'RZ450e Luxury Wagon', 'UX 200', 'UX 250h', 'UX 300e SUV'],
    'Wuling': ['Air EV Charging Pile Long Range Hatchback', 'Air EV Lite 200KM Hatchback', 'Air EV Lite 300KM Hatchback', 'Air EV Lite Hatchback', 'Air EV Lite Long Range Hatchback', 'Air EV Lite Standard Range Hatchback', 'Air EV Long Range + Charging Pile Hatchback', 'Air EV Long Range Hatchback', 'Air EV Pro 300KM Hatchback', 'Air EV Standard Range Hatchback', 'Almaz', 'Alvez', 'Binguo EV 333 Long Range AC Hatchback', 'Binguo EV 333 Long Range AC/DC Hatchback', 'Binguo EV 410 Premium Range AC/DC Hatchback', 'Binguo EV 7th Anniversary Premium Range Hatchback', 'Binguo EV Premium Range Hatchback', 'Cloud EV Hatchback', 'Cloud EV Lite Hatchback', 'Cloud EV Pro Hatchback', 'Confero', 'Cortez', 'Formo'],
    'BYD': ['Atto 3 Advanced SUV', 'Atto 3 Superior SUV', 'Dolphin Dynamic Standard Range Hatchback', 'Dolphin Premium Extended Range Hatchback', 'M6 EV Superior Captain Wagon', 'M6 EV Superior Wagon', 'M6 Standard 7-seaters MPV', 'M6 Superior 7-seaters MPV', 'M6 Superior Captain 6-seater MPV', 'Seal Performance AWD Sedan', 'Seal Premium Extended Range Sedan', 'Sealion 7 Performance SUV', 'Sealion 7 Premium SUV'],
    'Jeep': ['CJ-7', 'Cherokee', 'Compass', 'Gladiator', 'Grand Cherokee', 'Patriot', 'Wrangler'],
    'Peugeot': ['3008', '5008', 'RCZ'],
    'Dodge': ['Challenger', 'Journey'],
    'Chevrolet': ['Aveo', 'Captiva', 'Colorado', 'Express', 'Optra', 'Orlando', 'Spark', 'Spin', 'TRAX', 'Trailblazer', 'Trax'],
    'Isuzu': ['', 'D-Max', 'Elf', 'Giga', 'MU-X', 'Panther', 'Pickup', 'Traga', 'Trooper'],
    'Ford': ['Bronco', 'EcoSport', 'Everest', 'F-150', 'Fiesta', 'Focus', 'Mustang', 'Ranger'],
    'Land': ['Range Rover', 'Range Rover Evoque', 'Range Rover Sport', 'Range Rover Velar', 'Rover Defender', 'Rover Discovery', 'Rover Discovery 3', 'Rover Discovery 4', 'Rover Discovery Sport', 'Rover Hardtop Short'],
    'Subaru': ['BRZ', 'Crosstrek', 'Forester', 'Impreza', 'Legacy', 'WRX', 'WRX STi', 'XV'],
    'MG': ['4 EV Magnify i-SMART Hatchback', '5', 'Cyberster Convertible', 'HS', 'VS HEV', 'ZS', 'ZS EV Magnify SUV'],
    'Infiniti': ['FX37', 'Q50'],
    'Datsun': ['Cross', 'GO', 'GO+'],
    'smart': ['fortwo'],
    'Bentley': ['Bentayga', 'Continental', 'Continental Flying Spur', 'Continental GT', 'Continental Supersports'],
    'Morgan': ['Plus Four'],
    'Rolls-Royce': ['Ghost', 'Phantom', 'Wraith'],
    'Ferrari': ['296 GTB', '296 GTS', '458', '488 GTB', '488 Pista', '488 Spider', '612 Scaglietti', 'California', 'GTC4 Lusso T', 'GTC4Lusso T', 'Roma', 'SF90 Spider'],
    'Jaguar': ['F-Pace', 'S-Type', 'X-Type', 'XE', 'XF', 'XJ'],
    'Lamborghini': ['Aventador', 'Gallardo', 'Huracan'],
    'Genesis': ['G80 Electrified Sedan'],
    'Hummer': ['EV Edition 1 Wagon', 'H3'],
    'Volvo': ['S 80', 'XC40', 'XC60', 'XC90'],
    'Tesla': ['Model 3 Standard Range Plus Sedan', 'Model 3 Standard Range Sedan', 'Model S 100D Hatchback', 'Model S 100D Sedan', 'Model X 75D SUV', 'Model X Performance SUV', 'Model Y Long Range SUV', 'Model Y Standard SUV'],
    'GMC': ['Hummer EV EV 3X e4wd SUV', 'Hummer EV HEV Edition 1 SUV', 'Sierra', 'Yukon'],
    'DFSK': ['Gelora', 'Glory 560', 'Glory 580', 'Glory i-Auto', 'Super Cab'],
    'Cadillac': ['Escalade'],
    'GWM': ['Haval H6', 'Haval Jolion', 'Ora 03 Pro Hatchback', 'Tank'],
    'Maserati': ['Ghibli', 'GranTurismo', 'Levante', 'Quattroporte'],
    'Renault': ['Kiger', 'Koleos', 'Triber'],
    'Geely': ['EX 5 Max SUV', 'LC'],
    'McLaren': ['570S', '600LT', '650S', '720S', '765LT', 'MP4-12C', 'MP412C'],
    'Proton': ['Exora', 'Preve'],
    'Abarth': ['500', '595'],
    'Fiat': ['500', '500C'],
    'Alfa': ['Romeo 4C'],
    'GAC': ['AION V Luxury SUV', 'AION Y Plus 410 Exclusive MPV', 'AION Y Plus 490 Premium MPV', 'Hyptec HT Premium Hatchback'],
    'Hino': ['Dutro'],
    'RAM': ['1500'],
    'Ineos': ['Grenadier'],
    'Vinfast': ['VF 3 SUV'],
    'Denza': ['D9 Advanced MPV'],
    'Citroen': ['C 3', 'C3 Aircross', 'E-C3 Electric SUV'],
    'Aston': ['Martin Vantage'],
}

# Define default options for brands without specific mappings
for brand in brands:
    if brand not in car_type_mapping:
        car_type_mapping[brand] = ['Select Car Type']

# Define engine sizes for each car type
engine_size_mapping = {
    'City': ['1.5'],
    'Brio': ['1.2', '1.3'],
    'Cayman': ['2.7', '2.9', '3.4', '3.8'],
    'Grand Livina': ['1.5', '1.8', '2.5'],
    'Stargazer': ['1.5'],
    'Avanza': ['1.3', '1.5'],
    'Kijang Innova Zenix': ['2.0'],
    'Creta': ['1.5'],
    'Rush': ['1.5'],
    'Agya': ['1.0', '1.2'],
    'Alphard': ['2.4', '2.5', '3.0', '3.5'],
    'Ertiga': ['1.2', '1.4', '1.5'],
    'Pajero Sport': ['2.4', '2.5', '3.0'],
    'HR-V': ['1.5', '1.8'],
    'C 200': ['1.5', '1.8', '2.0'],
    'Calya': ['1.2'],
    'Xpander': ['1.5'],
    'Livina': ['1.5'],
    'Kijang Innova': ['2.0', '2.4', '2.5'],
    'Camry': ['2.4', '2.5', '3.5'],
    'Jazz': ['1.5'],
    '320i': ['2.0'],
    'C 250': ['1.8', '2.0'],
    'Ayla': ['1.0', '1.2'],
    'Santa Fe': ['1.6', '2.2', '2.4', '2.5'],
    'H-1': ['2.4', '2.5'],
    'WR-V': ['1.5'],
    'Civic': ['1.5', '1.6', '1.7', '1.8', '2.0'],
    'Mobilio': ['1.5'],
    'CX-5': ['2.0', '2.5'],
    'Sigra': ['1.0', '1.2'],
    'Odyssey': ['2.4'],
    'Raize': ['1.0', '1.2'],
    'CR-V': ['1.5', '2.0', '2.4'],
    'Ignis': ['1.2'],
    'A 6': ['2.0', '2.8'],
    'Sienta': ['1.5'],
    '2': ['1.5'],
    'March': ['1.2', '1.5'],
    'Xpander Cross': ['1.5'],
    'BR-V': ['1.5'],
    'Baleno': ['1.4', '1.5'],
    'Rio': ['1.4'],
    'Polo': ['1.2'],
    'Yaris': ['1.5', '1.6'],
    'Tiggo 7 Pro': ['1.5'],
    'Harrier': ['2.0', '2.4', '3.0', '3.5'],
    'Countryman': ['1.5', '1.6', '2.0'],
    'LX 570': ['5.7'],
    '330i': ['2.0', '3.0'],
    'Serena': ['1.4', '1.6', '2.0'],
    'Almaz': ['1.5', '2.0'],
    'X5': ['2.0', '3.0', '4.6'],
    'Seal Performance AWD Sedan': ['else'],
    '530i': ['2.0', '3.0'],
    'iX xDrive40 Sport SUV': ['else'],
    'Voxy': ['2.0'],
    'IONIQ 5 Signature Long Range SUV': ['else'],
    'Palisade': ['2.2'],
    'GLC 300': ['2.0'],
    'RX 200t': ['2.0'],
    'CX-3': ['1.5', '2.0'],
    'Vellfire': ['2.4', '2.5'],
    'CX-30': ['2.0'],
    'Air EV Long Range Hatchback': ['else'],
    'RX 270': ['2.7'],
    'Fortuner': ['2.4', '2.5', '2.7', '2.8'],
    'GLA 200': ['1.3', '1.6', '2.1'],
    'Cooper': ['1.5', '1.6', '2.0'],
    'Sidekick': ['1.6'],
    'Wrangler': ['2.0', '2.8', '3.0', '3.6', '3.8', '4.0'],
    '3008': ['1.6'],
    'Compass': ['1.4', '2.4'],
    'Journey': ['2.4'],
    'Vios': ['1.5'],
    'S350 L': ['3.0', '3.5'],
    '5008': ['1.6'],
    '520d': ['2.0'],
    'Elgrand': ['2.5', '3.5'],
    'ML250': ['2.1'],
    'Cortez': ['1.5', '1.8'],
    'Outlander Sport': ['2.0'],
    'Evalia': ['1.5'],
    'X1': ['1.5', '2.0'],
    'Juke': ['1.5'],
    'Etios Valco': ['1.2'],
    'Land Cruiser': ['2.8', '3.0', '3.3', '4.2', '4.5', '4.6', '4.7'],
    'X-Trail': ['2.0', '2.5'],
    'Binguo EV 410 Premium Range AC/DC Hatchback': ['else'],
    'Xenia': ['1.0', '1.3', '1.5'],
    'Terios': ['1.5'],
    'Swift': ['1.4', '1.5', '1.6'],
    'Captiva': ['2.0', '2.4'],
    'B200': ['1.3', '1.6'],
    'Veloz': ['1.5'],
    'SX4 S-Cross': ['1.5'],
    'GLC 200': ['2.0'],
    'Freed': ['1.5'],
    'Corolla Altis': ['1.8', '2.0'],
    'Golf': ['1.2', '1.4', '2.0'],
    'TRAX': ['1.4'],
    'Accord': ['1.5', '2.0', '2.2', '2.3', '2.4'],
    'Taft': ['2.8'],
    'Grand Vitara': ['1.5', '2.0', '2.4'],
    'GLE 400': ['3.0'],
    'ES 250': ['2.5'],
    'E 300': ['2.0', '3.0'],
    'Alvez': ['1.5'],
    'XL7': ['1.5'],
    'CLA200': ['1.3', '1.6'],
    'GLE 450': ['3.0'],
    'Macan': ['2.0', '2.9', '3.0'],
    'Panther': ['2.2', '2.5'],
    'E 350': ['2.0', '3.5'],
    '86': ['2.0'],
    '218i': ['1.5'],
    'S-Presso': ['1.0'],
    'C200K': ['1.8'],
    'Fiesta': ['1.4', '1.5', '1.6'],
    'Hiace': ['2.5', '2.8', '3.0'],
    'Delica': ['2.0'],
    'Gran Max': ['1.3', '1.5'],
    'S 400L': ['3.0'],
    'C 300': ['2.0', '3.0'],
    'GLC 250': ['2.0'],
    'Teana': ['2.0', '2.3', '2.5'],
    'CLS 350': ['2.0', '3.5'],
    'Picanto': ['1.0', '1.1', '1.2', '1.3'],
    'GR Yaris': ['1.6'],
    'Confero': ['1.5'],
    'EcoSport': ['1.5'],
    'M3': ['3.0', '4.0'],
    'Range Rover Velar': ['2.0', '3.0'],
    'ML350': ['3.5'],
    'Carry': ['0.5', '1.5'],
    '328i': ['2.0'],
    '730Li': ['2.0', '3.0'],
    'BRZ': ['2.0', '2.4'],
    'Range Rover Evoque': ['2.0'],
    'C 230': ['2.3', '2.5'],
    'Omoda E5 EV Pure SUV': ['else'],
    'MX-5': ['2.0'],
    'M4': ['3.0'],
    'NAV1': ['2.0'],
    'Range Rover Sport': ['2.0', '3.0', '3.6', '4.2', '5.0'],
    'Biante': ['2.0'],
    'Sportage': ['2.0'],
    'Atto 3 Superior SUV': ['else'],
    'HS': ['1.5'],
    'A 4': ['1.8', '2.0'],
    'Crosstrek': ['2.0'],
    'Yaris Cross': ['1.5'],
    'Tucson': ['1.6', '2.0'],
    'Sonet': ['1.5'],
    'LS 460L': ['4.6'],
    '718': ['2.0', '2.5', '4.0'],
    'ML400': ['3.0'],
    'X7': ['3.0'],
    'FX37': ['3.7'],
    'RX 300': ['2.0'],
    '520i': ['2.0', '2.2'],
    'X3': ['2.0', '3.0'],
    'Tiguan': ['1.4'],
    'Karimun Wagon R': ['1.0'],
    'C 180': ['1.5', '1.8', '2.0'],
    'Splash': ['1.2'],
    'Kicks': ['1.2'],
    'GO': ['1.2'],
    'CX-9': ['2.5', '3.7'],
    '': ['1.6', '2.5', '2.7'],
    'Scirocco': ['1.4', '2.0'],
    'C-HR': ['1.8'],
    'Pride': ['1.4'],
    'Omoda 5': ['1.5', '1.6'],
    'Luxio': ['1.5'],
    'V260': ['2.0'],
    'Rocky': ['1.0', '1.2'],
    'i7 xDrive60 Gran Lusso Sedan': ['else'],
    'Katana': ['1.0'],
    'EV6 GT-Line SUV': ['else'],
    '911': ['3.0', '3.4', '3.8', '4.0'],
    'LM 350': ['3.5'],
    'fortwo': ['1.0'],
    'Sirion': ['1.3'],
    'Grand Carnival': ['2.2'],
    'ES 300h': ['2.5'],
    'GL400': ['3.0'],
    'SX4': ['1.5'],
    'GO+': ['1.2'],
    'E 250': ['1.8', '2.0', '2.1'],
    'RX 350': ['2.4', '3.5'],
    'MU-X': ['2.5'],
    'Q3': ['1.4', '2.0'],
    'GLS 450': ['2.9', '3.0'],
    '435i': ['3.0'],
    'Caravelle': ['2.0'],
    'C 240': ['2.6'],
    'Range Rover': ['2.0', '3.0', '4.2', '4.4', '5.0'],
    'CLS 400': ['3.0'],
    'Jimny': ['1.3', '1.5'],
    'EQB250 Progressive Line SUV': ['else'],
    '3': ['1.6', '2.0'],
    'Triton': ['2.4', '2.5'],
    'Hilux': ['2.0', '2.4', '2.5'],
    'Rover Defender': ['2.0', '2.2', '2.3', '3.0'],
    'Z4': ['2.0', '2.5', '3.0'],
    'Rover Discovery 4': ['3.0'],
    'Corolla Cross': ['1.8'],
    'Colt L300': ['2.3', '2.5'],
    'Continental Flying Spur': ['6.0'],
    'Paceman': ['1.6'],
    'Roadster': ['1.6'],
    'Cabrio': ['1.5', '1.6', '2.0'],
    'SPRINTER': ['2.1'],
    'V250': ['2.0', '2.1'],
    'Mirage': ['1.2'],
    'Murano': ['2.5', '3.5'],
    'S500 L': ['3.0', '4.7', '5.0', '5.5'],
    'S 450L': ['3.0'],
    'SL300': ['3.0'],
    'Outlander': ['2.4'],
    'SL500': ['5.0'],
    'Lancer': ['1.6', '1.8'],
    'Cayenne': ['3.0', '3.2', '3.6', '4.0', '4.5', '4.8'],
    'Boxster': ['2.7', '2.9'],
    'Strada Triton': ['2.5', '2.8'],
    'Plus Four': ['2.0'],
    'UX 200': ['2.0'],
    'AMG CLA45': ['2.0'],
    'E 400': ['3.0'],
    'G55 AMG': ['5.4'],
    'E200K': ['1.8'],
    'G63 AMG': ['4.0', '5.5'],
    'EQS450+ Electric Art Line Sedan': ['else'],
    'E 200': ['1.8', '2.0'],
    'G 300': ['3.0'],
    'XFORCE': ['1.5'],
    'SL350': ['3.5'],
    'S 400': ['3.5'],
    'Terra': ['2.5'],
    'R280L': ['3.0'],
    'S 320': ['3.2'],
    'Clubman': ['2.0'],
    'E 280': ['3.0'],
    'Kijang': ['1.5', '1.8', '2.0', '2.4'],
    'Stream': ['1.7', '2.0'],
    'Wraith': ['6.6'],
    'CX-8': ['2.5'],
    'California': ['3.9', '4.3'],
    'E53 AMG': ['3.0'],
    'Maybach S500': ['4.7'],
    'GLB200': ['1.3'],
    'E 320': ['3.2'],
    'Terrano': ['2.4'],
    'CX-60': ['2.5', '3.3'],
    'CL500': ['5.0', '5.5'],
    'E63 AMG': ['5.5', '6.2'],
    'GL63': ['5.5'],
    'XJ': ['2.0', '3.0', '5.0'],
    'Patriot': ['2.4'],
    'RX 350h': ['2.5'],
    'Sorento': ['2.2', '2.4'],
    'XE': ['2.0'],
    'RX 500H': ['2.4'],
    'SLC200': ['2.0'],
    'SL400': ['3.0'],
    'A35 AMG': ['2.0'],
    'Gallardo': ['5.0', '5.2'],
    'Mustang': ['2.3', '5.0'],
    'Magnite': ['1.0'],
    'AMG GLE53': ['3.0'],
    'IST': ['1.5'],
    'NBeetle': ['2.0'],
    'CLK230K': ['2.3'],
    'Gls400': ['3.0'],
    'AMG GLA35': ['2.0'],
    'LM 350h': ['2.5', '3.5'],
    'G 400': ['2.9'],
    'Grand Sedona': ['2.2', '3.3'],
    'NX 200T': ['2.0'],
    'IONIQ 5 Prime Long Range SUV': ['else'],
    'APV': ['1.5'],
    'Panamera': ['2.9', '3.0', '3.6', '4.8'],
    'NX 300': ['2.0'],
    'X6': ['3.0', '4.4'],
    'Karimun': ['1.0'],
    'Staria': ['2.2'],
    'IONIQ 6 Fastback': ['else'],
    'G80 Electrified Sedan': ['else'],
    'SLK250': ['1.8'],
    'M2': ['3.0'],
    'E43 AMG': ['3.0', '3.2'],
    'H3': ['3.7'],
    'SLK200': ['1.8', '2.0'],
    'Cloud EV Hatchback': ['else'],
    'AMG GLA45': ['2.0'],
    'XC40': ['1.5', '2.0'],
    'A 200': ['1.3', '1.6'],
    'XV': ['2.0'],
    'GR Corolla': ['1.6'],
    'AMG GLE43': ['3.0'],
    'Cooper S Electric Resolute Edition 3 Door Hatchback': ['else'],
    'GL500': ['4.7', '5.5'],
    'Pajero': ['2.5', '3.2'],
    'Land Cruiser Cygnus': ['4.7'],
    'C 280': ['3.0'],
    'Land Cruiser Prado': ['2.7', '2.8'],
    'Colt T120SS': ['1.5'],
    'Model Y Long Range SUV': ['else'],
    'AMG CLS63': ['2.0', '5.5'],
    'Hummer EV EV 3X e4wd SUV': ['else'],
    '318i': ['1.8', '1.9', '2.0'],
    'Bentayga': ['6.0'],
    'Omoda E5 EV SUV': ['else'],
    'M5': ['4.4'],
    'Huracan': ['5.2'],
    '740Li': ['3.0'],
    'C43 AMG': ['3.0'],
    'S 660': ['0.7'],
    'Air EV Long Range + Charging Pile Hatchback': ['else'],
    'Stargazer X': ['1.5'],
    '735i': ['3.0'],
    'Continental GT': ['4.0', '6.0'],
    'Q8': ['3.0'],
    'GLE 250': ['2.1'],
    '430i': ['2.0'],
    'Glory 580': ['1.5'],
    'Taycan 4S Performance Battery Sedan': ['else'],
    'Q7': ['3.0'],
    'M235i': ['2.0', '3.0'],
    '550i': ['4.4'],
    'WRX STi': ['2.5'],
    'Escalade': ['6.2'],
    'Trailblazer': ['2.4', '2.5'],
    '300CE': ['3.0'],
    'Tank': ['2.0'],
    'R8': ['4.2', '5.2'],
    'Colt': ['1.3', '3.9'],
    'Seltos': ['1.4', '1.5'],
    'Cross': ['1.2'],
    'Eclipse Cross': ['1.5'],
    'Spark': ['1.4'],
    'Ghibli': ['3.0'],
    'ZS': ['1.5'],
    '645Ci': ['4.4'],
    'LX 600': ['3.4'],
    '840i': ['3.0'],
    'S-Type': ['3.0'],
    '6': ['2.5'],
    'i4 eDrive40 M Sport Coupe': ['else'],
    'RS4': ['2.9'],
    '535i': ['3.0'],
    'i4 eDrive35 Coupe': ['else'],
    'i5 eDrive40 M Sport Sedan': ['else'],
    'Formo': ['1.2'],
    'Elysion': ['3.0'],
    'Orlando': ['1.8'],
    'Colt Diesel': ['3.3', '3.9', '4.2'],
    '528i': ['2.0', '2.8', '3.0'],
    'Focus': ['1.6', '1.8', '2.0'],
    'Cherokee': ['2.4', '4.0'],
    'BZ4X (1 Tone) SUV': ['else'],
    'RCZ': ['1.6', '2.0'],
    '320d': ['2.0'],
    'TT': ['2.0'],
    'Koleos': ['2.5'],
    'M135i': ['2.0', '3.0'],
    'Binguo EV 333 Long Range AC Hatchback': ['else'],
    'S 300L': ['3.0'],
    '325i': ['2.5'],
    'GR 86': ['2.4'],
    'Model 3 Standard Range Sedan': ['else'],
    'Navara': ['2.5'],
    'RX-8': ['1.3'],
    'Getz': ['1.3'],
    'Impreza': ['1.6', '2.0'],
    'Escudo': ['1.6', '2.0'],
    'Frontier Navara': ['2.5'],
    'Corolla': ['1.6', '1.8'],
    'S 280': ['2.8'],
    'V220': ['2.1'],
    '760Li': ['6.0'],
    'Etios': ['1.2', '1.5'],
    'IONIQ 5 Prime Standard Range SUV': ['else'],
    'Carens': ['1.4', '1.5', '1.8'],
    'Seal Premium Extended Range Sedan': ['else'],
    'GranTurismo': ['4.7'],
    'Taycan Performance Battery Sedan': ['else'],
    'Elf': ['2.8', '3.0', '4.8'],
    'C 350': ['3.5'],
    'TBeetle': ['1.4'],
    '458': ['4.5'],
    'iX1 eDrive20 M Sport SUV': ['else'],
    'GR Supra': ['3.0'],
    'UX 300e SUV': ['else'],
    'A 8': ['3.0'],
    'A 3': ['1.2', '2.0'],
    'LC': ['1.3'],
    '612 Scaglietti': ['5.7'],
    'XC60': ['2.0'],
    'CR-Z': ['1.5'],
    'Air EV Standard Range Hatchback': ['else'],
    'Carnival': ['2.2', '2.9'],
    '570S': ['2.5', '3.8'],
    'Preve': ['1.6'],
    'AMG GT': ['3.0', '4.0'],
    '640i': ['3.0'],
    'Model X 75D SUV': ['else'],
    'Spin': ['1.2', '1.5'],
    'Triber': ['1.0'],
    'iQ': ['1.0'],
    'E 230': ['2.3', '2.5'],
    'Tiggo 8 Pro': ['2.0'],
    'Kona': ['2.0'],
    'Starlet': ['1.3'],
    '488 Spider': ['3.9'],
    'GTC4 Lusso T': ['3.9'],
    '488 Pista': ['3.9'],
    '600LT': ['3.0', '3.8'],
    'Model S 100D Sedan': ['else'],
    '650S': ['3.8'],
    '765LT': ['4.0'],
    'Continental Supersports': ['6.0'],
    'Kuda': ['1.6', '2.5'],
    'Model 3 Standard Range Plus Sedan': ['else'],
    'Model X Performance SUV': ['else'],
    'Taycan 4S Performance Battery Plus Sedan': ['else'],
    'Taycan 4S Cross Turismo Coupe': ['else'],
    'Kona EV SUV': ['else'],
    'EV9 GT-Line SUV': ['else'],
    '116i': ['1.6'],
    'Air EV Lite Hatchback': ['else'],
    'AMG SL43': ['2.0', '4.0'],
    'Q50': ['2.0'],
    'Glory 560': ['1.5'],
    '8': ['2.3'],
    'Celica': ['1.8'],
    'Traga': ['2.5'],
    'CX-7': ['2.3'],
    'RX 450h+': ['2.5'],
    'RS5': ['2.9'],
    'Sonet 7': ['1.5'],
    'IONIQ Signature Coupe': ['else'],
    'Mega Carry': ['1.5'],
    'VS HEV': ['1.5'],
    'GS 200t': ['2.0'],
    'Everest': ['2.0', '2.2', '2.5'],
    'A 250': ['2.0'],
    'Kona Signature SUV': ['else'],
    'Glory i-Auto': ['1.5'],
    'A 1': ['1.4'],
    'Q5': ['2.0'],
    'M6': ['4.4'],
    '323i': ['2.5'],
    'Markx': ['2.5'],
    '630i': ['3.0'],
    'Hummer EV HEV Edition 1 SUV': ['else'],
    '488 GTB': ['3.9'],
    'AMG SLC43': ['3.0'],
    'XC90': ['2.0'],
    'G 350': ['3.0'],
    'GranAce': ['2.8'],
    'X-Type': ['2.1'],
    'A 5': ['2.0', '3.2'],
    'Zebra': ['1.3'],
    '335i': ['3.0'],
    'S 80': ['3.2'],
    'XF': ['2.0', '3.0'],
    'Tanto': ['0.7'],
    'Super Cab': ['1.5'],
    'GS 300': ['3.0'],
    'WRX': ['2.4'],
    'A45 AMG': ['2.0'],
    'Trooper': ['3.1'],
    '750Li': ['4.4'],
    'Previa': ['2.4'],
    'Galant': ['2.5'],
    'Ghost': ['6.6'],
    '500': ['1.4'],
    'Taruna': ['1.5', '1.6'],
    'SLK200K': ['1.8'],
    '428i': ['2.0'],
    '220i': ['2.0'],
    '523i': ['2.5'],
    'Ranger': ['2.0', '2.2', '2.5', '3.2'],
    'Rover Discovery Sport': ['2.0'],
    'Crown': ['2.0', '2.4', '2.5', '2.8', '3.0'],
    'Legacy': ['2.0'],
    '720S': ['4.0'],
    'Aventador': ['6.5'],
    'MP412C': ['3.8'],
    'Travello': ['2.7'],
    'ML320': ['3.2'],
    'Avega': ['1.5'],
    'RZ 450E Luxury SUV': ['else'],
    '370Z': ['2.0'],
    'T Cross': ['1.0'],
    'F-Pace': ['3.0'],
    'Rover Hardtop Short': ['2.3'],
    'EQS450 AMG Line 4MATIC SUV': ['else'],
    'FJ Cruiser': ['4.0'],
    'Cooper S Electric Level 3 Hatchback': ['else'],
    'Vito': ['2.0'],
    'X4': ['2.0', '3.0'],
    'i8': ['1.5'],
    'Quattroporte': ['3.0'],
    'Roma': ['3.9'],
    'M850i': ['4.4'],
    'LS 500': ['3.4'],
    'Rover Discovery': ['3.0'],
    'Levante': ['3.0'],
    'S 600L': ['5.5'],
    'LS 600hL': ['5.0'],
    'RC F': ['5.0'],
    'Gelora': ['1.5'],
    'SLK230': ['2.3'],
    'Trajet': ['2.0'],
    'EQE350+ Electric Art Line Sedan': ['else'],
    'S 600': ['6.0'],
    'Zebra Espass': ['1.3'],
    'Esteem': ['1.3'],
    'Excel': ['1.5'],
    'E 220': ['2.2'],
    'CT 200h': ['1.8'],
    'Ciaz': ['1.4'],
    'Colorado': ['2.4'],
    'BZ4X (2 Tone) SUV': ['else'],
    'Grand i10': ['1.2'],
    'CJ-7': ['4.2'],
    'Grand Cherokee': ['3.0', '3.6', '4.0', '5.7', '6.4'],
    'Taycan Turbo Sedan': ['else'],
    'Noah': ['2.0'],
    'Dyna': ['4.0'],
    'E 240': ['2.4', '2.6'],
    'CLS 500': ['5.0'],
    'Touareg': ['3.0'],
    'Rover Discovery 3': ['2.7'],
    'X2': ['1.5'],
    '280GE': ['2.7'],
    '730Ld': ['3.0'],
    'Express': ['5.3'],
    'i20': ['1.4'],
    'Grand Avega': ['1.4'],
    'Phantom': ['6.7'],
    'Romeo 4C': ['1.7'],
    'RX 330': ['3.3'],
    'Grandis': ['2.4'],
    'Soluna': ['1.5'],
    'D-Max': ['1.9'],
    'Optra': ['1.6'],
    'Frontier': ['3.0'],
    'Limo': ['1.5'],
    'Prelude': ['2.2'],
    'Touran': ['1.4'],
    'Carry Real': ['1.5'],
    '5': ['1.5'],
    'Haval Jolion': ['1.5'],
    'UX 250h': ['2.0'],
    'Haval H6': ['1.5'],
    'Air EV Lite Long Range Hatchback': ['else'],
    'Aceman SE Hatchback': ['else'],
    'IONIQ 5 Batik Edition SUV': ['else'],
    'Tiggo 8': ['1.6'],
    'Tiggo Cross': ['1.5'],
    'EQS450+ Edition One Sedan': ['else'],
    'G 500': ['3.0'],
    'Sealion 7 Premium SUV': ['else'],
    'M6 Superior 7-seaters MPV': ['else'],
    'Dolphin Dynamic Standard Range Hatchback': ['else'],
    'AION Y Plus 410 Exclusive MPV': ['else'],
    'Omoda J6 EV IWD SUV': ['else'],
    'Kona Signature Long Range SUV': ['else'],
    'Kona Prime Standard Range SUV': ['else'],
    'Feroza': ['1.6'],
    'M6 Standard 7-seaters MPV': ['else'],
    'Dolphin Premium Extended Range Hatchback': ['else'],
    'Atto 3 Advanced SUV': ['else'],
    'Binguo EV 333 Long Range AC/DC Hatchback': ['else'],
    'Air EV Pro 300KM Hatchback': ['else'],
    'Sealion 7 Performance SUV': ['else'],
    'Venue': ['1.5'],
    'M6 Superior Captain 6-seater MPV': ['else'],
    'ID. Buzz Style Van': ['else'],
    'Hilux Rangga': ['2.0', '2.4'],
    'Fuso Canter': ['3.9'],
    'Cloud EV Pro Hatchback': ['else'],
    'Dutro': ['4.0'],
    'Cloud EV Lite Hatchback': ['else'],
    'Maybach S580': ['4.0'],
    '4 EV Magnify i-SMART Hatchback': ['else'],
    '1500': ['6.2'],
    'Grenadier': ['3.0'],
    'Omoda J6 EV RWD SUV': ['else'],
    'Air EV Lite 300KM Hatchback': ['else'],
    'Sierra': ['6.2'],
    'ID. Buzz Life Van': ['else'],
    'CX-80': ['2.5'],
    'Omoda J6 EV IWD Phantom Edition SUV': ['else'],
    'Air EV Lite 200KM Hatchback': ['else'],
    'Kona N-Line SUV': ['else'],
    '595': ['1.4'],
    '296 GTS': ['3.0'],
    'C63 AMG': ['2.0'],
    'Omoda J6 EV RWD Phantom Edition SUV': ['else'],
    'VF 3 SUV': ['else'],
    'D9 Advanced MPV': ['else'],
    'Tiggo 5x': ['1.5'],
    'ML63': ['5.5'],
    'Maybach S560': ['4.0'],
    'G 450': ['3.0'],
    'NX 350h': ['2.5'],
    'Prius': ['2.0'],
    'Model Y Standard SUV': ['else'],
    'CLE 300': ['2.0'],
    'Kona Signature Standard Range SUV': ['else'],
    'Forester': ['2.0'],
    'RAV4': ['2.5'],
    '300': ['3.0'],
    'H100': ['2.6'],
    'Cooper Cooper SE Hatchback': ['else'],
    '190E': ['2.0'],
    'XM': ['3.0', '4.4'],
    'Air EV Lite Standard Range Hatchback': ['else'],
    'Countryman SE ALL4 SUV': ['else'],
    'IONIQ 5 Signature Standard Range SUV': ['else'],
    'EX 5 Max SUV': ['else'],
    'ZS EV Magnify SUV': ['else'],
    'G580 EQ SUV': ['else'],
    'Z': ['3.0'],
    'Kona Prime Long Range SUV': ['else'],
    'F-150': ['3.5'],
    'Ora 03 Pro Hatchback': ['else'],
    'Binguo EV 7th Anniversary Premium Range Hatchback': ['else'],
    'Bronco': ['2.7'],
    'M8': ['4.4'],
    'Leaf (1 Tone) Hatchback': ['else'],
    'C 3': ['1.2'],
    'C3 Aircross': ['1.2'],
    'E-C3 Electric SUV': ['else'],
    'Hyptec HT Premium Hatchback': ['else'],
    'AION V Luxury SUV': ['else'],
    'AION Y Plus 490 Premium MPV': ['else'],
    '296 GTB': ['3.0'],
    'Cyberster Convertible': ['else'],
    'Yukon': ['6.2'],
    'Challenger': ['6.2'],
    'RS6': ['4.0'],
    '1 Series M': ['3.0'],
    'Gladiator': ['3.6'],
    'S 350': ['3.7'],
    'GX 550': ['3.4'],
    '500C': ['1.4'],
    'Kiger': ['1.0'],
    'Giga': ['5.2'],
    'S450 L': ['3.0'],
    'LM350h': ['2.5'],
    'IONIQ 6 Signature Long Range Sedan': ['else'],
    'GLE 53': ['3.0'],
    'Trax': ['1.4'],
    'Innova Venturer': ['2.0', '2.4'],
    'IONIQ 5 Long Range Signature SUV': ['else'],
    'BZ4X Wagon': ['else'],
    'LM350': ['3.5'],
    'GLS 400': ['3.0'],
    'S400 L': ['3.0'],
    'GLE 43': ['3.0'],
    'EV Edition 1 Wagon': ['else'],
    'Omoda E5 EV Wagon': ['else'],
    'Livina X-Gear': ['1.5'],
    'GLA 45': ['2.0'],
    'Estima': ['2.4'],
    'CLA45 AMG': ['2.0'],
    'SF90 Spider': ['4.0'],
    'Air EV Charging Pile Long Range Hatchback': ['else'],
    'Continental': ['6.0'],
    'EQS 450 Electric 4MATIC AMG Line Wagon': ['else'],
    'Amenity': ['1.3'],
    'Kona Signature Wagon': ['else'],
    'S300 L': ['3.0'],
    'i4 eDrive40 M Sport Gran Coupe': ['else'],
    'i4 eDrive35 Gran Coupe': ['else'],
    'NX 200t': ['2.0'],
    'Exora': ['1.6'],
    'T-Cross': ['1.0'],
    'Sprinter': ['2.1'],
    'Camry Hybrid': ['2.5'],
    'IONIQ Electric Signature Fastback': ['else'],
    'GTC4Lusso T': ['3.9'],
    'Model S 100D Hatchback': ['else'],
    'MP4-12C': ['3.8'],
    'B170': ['1.7'],
    'Binguo EV Premium Range Hatchback': ['else'],
    'Every': ['1.3'],
    'Pickup': ['2.5'],
    'iX xDrive40 Sport Wagon': ['else'],
    'Canter': ['3.9'],
    'Patrol': ['4.2'],
    'Aveo': ['1.4'],
    'M6 EV Superior Captain Wagon': ['else'],
    'RZ450e Luxury Wagon': ['else'],
    'M6 EV Superior Wagon': ['else'],
    'Martin Vantage': ['4.0']
    }

# For car types not explicitly defined
for car_type in [car_type for brand in car_type_mapping for car_type in car_type_mapping[brand]]:
    if car_type not in engine_size_mapping and car_type != 'Select Car Type':
        engine_size_mapping[car_type] = ['Default']

# Functions to get dropdown options
def get_car_types(brand):
    """Get car types for the selected brand"""
    return car_type_mapping.get(brand, ['Select Car Type'])

def get_engine_sizes(car_type):
    """Get engine sizes for the selected car type"""
    return engine_size_mapping.get(car_type, ['Select Engine Size'])

# Function to prepare input for model prediction
def prepare_input_data(location, brand, car_type, machine_type, engine_size, kilometer, year):
    """
    Process input data to match the model's expected features.
    """
    # Create a dataframe with a single row
    input_data = pd.DataFrame({
        'location': [location],
        'brand': [brand],
        'brand_type': [f"{car_type} {engine_size}"],
        'transmission': [machine_type],
        'KM_1': [kilometer],
        'year': [year]
    })
    
    # Feature engineering to match the model's expected input
    # Location features
    input_data['is_dki_jakarta'] = (input_data['location'] == 'DKI Jakarta').astype(int)
    input_data['is_jawa'] = input_data['location'].isin(['Banten', 'Jawa Barat', 'Jawa Timur', 'Jawa Tengah', 'Yogyakarta']).astype(int)
    
    # Transmission feature
    input_data['is_automatic'] = (input_data['transmission'] == 'Automatic').astype(int)
    
    # Brand country features
    input_data['is_japan'] = input_data['brand'].isin(japan_brands).astype(int)
    input_data['is_korea'] = input_data['brand'].isin(korea_brands).astype(int)
    input_data['is_germany'] = input_data['brand'].isin(germany_brands).astype(int)
    input_data['is_USA'] = input_data['brand'].isin(usa_brands).astype(int)
    input_data['is_china'] = input_data['brand'].isin(china_brands).astype(int)
    input_data['is_italy'] = input_data['brand'].isin(italy_brands).astype(int)
    
    # Luxury and sport car features
    input_data['is_luxury'] = (
        input_data['brand'].isin(luxury_brands) | 
        (input_data['brand_type'] == 'Civic 2.0 Type R Hatchback') |
        (input_data['brand_type'].str.contains('Land Cruiser', na=False)) |
        (input_data['brand_type'].str.contains('Mustang', na=False))
    ).astype(int)
    
    input_data['is_sport_car'] = input_data['brand'].isin(sport_car_brands).astype(int)
    
    # Extract type and engine size
    def extract_type_and_engine(text):
        if pd.isna(text):
            return pd.Series([None, None])
        # Search for first number pattern like 1.2, 2.5, etc.
        match = re.search(r'(\d\.\d)', text)
        if match:
            engine = float(match.group(1))
            before_engine = text[:match.start()].strip()
            return pd.Series([str(before_engine), float(engine)])
        else:
            return pd.Series([text.strip(), None])
    
    # Apply to brand_type column
    input_data[['type_', 'engine_size']] = input_data['brand_type'].apply(extract_type_and_engine)
    input_data['engine_size_str'] = input_data['engine_size'].map(lambda x: f"{x:.1f}" if pd.notnull(x) else None)
    
    # For simplicity, we'll use placeholder values for encoded features
    # In a real application, you would need to use the same label encoder used during training
    input_data['type_fix'] = input_data['type_']
    input_data['type_fix_encoded'] = 0  # Placeholder
    input_data['engine_size_str'] = input_data['engine_size_str'].fillna('else')
    input_data['engine_size_str_encoded'] = 0  # Placeholder
    
    # Log transformations
    input_data['ln_KM_1'] = np.log(input_data['KM_1'])
    
    # Select the required features for the model
    features = [
        'is_jawa',
        'is_automatic',
        'is_japan',
        'is_korea',
        'is_germany',
        'is_USA',
        'is_china',
        'is_italy',
        'is_luxury',
        'is_sport_car',
        'year',
        'ln_KM_1',
        'type_fix_encoded',
        'engine_size_str_encoded'
    ]
    
    return input_data[features]

# Create the sidebar for input parameters
with st.sidebar:
    st.header("Car Details")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    # Location selection
    with col1:
        location = st.selectbox("Location", locations)
    
    # Brand selection
    with col2:
        brand = st.selectbox("Brand", brands)
    
    # Car type selection (dependent on brand)
    car_types = get_car_types(brand)
    car_type = st.selectbox("Car Type", car_types)
    
    # Transmission type selection (always Automatic and Manual)
    machine_type = st.selectbox("Transmission Type", ['Automatic', 'Manual'])
    
    # Engine size selection (dependent on car type)
    engine_sizes = get_engine_sizes(car_type)
    engine_size = st.selectbox("Engine Size", engine_sizes)
    
    # Numeric inputs
    car_kilometer = st.number_input("Car Kilometer", min_value=1, max_value=1000000, step=1000, value=10000)
    car_year = st.number_input("Car Year", min_value=1990, max_value=2025, value=2020)
    
    # Prediction button
    predict_button = st.button("Predict Price")

# Main area for results
if predict_button:
    # Check if all required fields are filled
    if (location != '' and brand != '' and car_type != 'Select Car Type' and 
        machine_type in ['Automatic', 'Manual'] and engine_size != 'Select Engine Size'):
        
        # Prepare the input data for prediction
        input_data = prepare_input_data(
            location, brand, car_type, machine_type, engine_size, car_kilometer, car_year
        )
        
        
        try:
                # Make prediction
                #prediction = model.predict(input_data)
                request_body = {
                    'location': location,   
                    'brand': brand,
                    'car_type': car_type, 
                    'machine_type': machine_type,   
                    'engine_size': engine_size,
                    'kilometer': car_kilometer,
                    'year': car_year
                }
                prediction = requests.post("https://apiavm.rhr.co.id/cars/predict/", json = request_body, verify=False)
                print(prediction)
                
                # Display prediction
                st.success(f"Predicted Car Price: Rp {prediction.json()['price']}")
                
                # Display the input features that were used
                st.subheader("Car Details")
                st.write(f"**Location:** {location}")
                st.write(f"**Brand:** {brand}")
                st.write(f"**Car Type:** {car_type}")
                st.write(f"**Transmission Type:** {machine_type}")
                st.write(f"**Engine Size:** {engine_size}")
                st.write(f"**Kilometer:** {car_kilometer:,}")
                st.write(f"**Year:** {car_year}")
                

        except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please check model compatibility with the input data or adjust the code as needed.")
    else:
        st.warning("Please fill in all the required fields")

# Add instructions at the bottom
with st.expander("How to use this app"):
    st.write("""
    1. Select the location where the car is being sold
    2. Choose the car brand
    3. Select the car type (model) for the chosen brand
    4. Choose the transmission type (Automatic or Manual)
    5. Select the engine size available for this car type
    6. Enter the car's kilometer reading
    7. Enter the car's manufacturing year
    8. Click the 'Predict Price' button to get the estimated price
    """)

# Add information about the model at the bottom
with st.expander("About the Model"):
    st.write("""
    This app uses a Random Forest Regressor model trained on historical car price data.
    The model takes into account various factors like the car's brand origin, location, 
    engine specifications, transmission type, mileage, and age to estimate its market price.
    
    Key features used by the model include:
    - Location (Jawa vs DKI Jakarta vs others)
    - Brand country of origin (Japan, Korea, Germany, USA, China, Italy)
    - Car type and engine size
    - Transmission type (automatic vs manual)
    - Luxury and sports car status
    - Year and kilometer reading
    
    Note: This is a predictive model and actual market prices may vary.
    """)