from storagemanager import  StorageManager


# Class -  group

# Object -

sm = StorageManager()


            
def send_data(studentFileLocalPath, professionalFileLocalPath, imgNum, current_date_time):


    public_url1 = sm.upload_file(file_name=f"dance/{current_date_time}/student{imgNum}" , local_path=studentFileLocalPath)
    public_url2 = sm.upload_file(file_name=f"dance/{current_date_time}/professional{imgNum}" , local_path=professionalFileLocalPath)

    public_urls = (public_url1, public_url2)

    return public_urls

# # blueprint recipe
# class Dog:

#     def __init__(self, name, age, breed) -> None:
        
#         self.name = name
#         self.age = age
#         self.breed = breed
#         self.numFeet = 4
    
#     def changeName(self, newName):

#         self.name = newName



# # OBJECT

# dog1 = Dog("Buster", 2, "Pitbull")  # __init__

# print(dog1.name)

# dog1.changeName("Kevin")

# print(dog1.name)