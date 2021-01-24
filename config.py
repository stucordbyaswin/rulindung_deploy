
# base config
class Config(object):
    DEBUG = False
    TESTING = False

    

# seluruh kelas di bawah ini akan meng-inherit kelas "Config" yang di atas. 
class ProductionConfig(Config):
    DEBUG = False
    IMAGE_UPLOAD_PATH = "C:/Users/Aswin/Desktop/surviviol_id/app/static/img/uploads"
    IMAGE_ACCESS_PATH = "../static/img/uploads"
    ALLOWED_IMAGES_EXTENSIONS = ['JPG', 'JPEG', 'PNG', 'GIF']
    MAX_IMAGE_SIZE = 0.5 * 1024 * 1024

    MODEL_1_PATH = "C:/Users/Aswin/Desktop/surviviol_id/model/normalized/" # Model with normalized dataset
    MODEL_2_PATH = "C:/Users/Aswin/Desktop/surviviol_id/model/non_normalized/" # Model with normalized dataset

class DevelopmentConfig(Config):
    DEBUG = True
    IMAGE_UPLOAD_PATH = "/var/www/rulindung/app/static/img/uploads"
    IMAGE_ACCESS_PATH = "../static/img/uploads"
    ALLOWED_IMAGES_EXTENSIONS = ['JPG', 'JPEG', 'PNG', 'GIF']
    MAX_IMAGE_SIZE = 0.5 * 1024 * 1024

    MODEL_1_PATH = "/var/www/rulindung/model/normalized/" # Model with normalized dataset
    MODEL_2_PATH = "/var/www/rulindung/model/non_normalized/" # Model with normalized dataset
class TestingConfig(Config):
    DEBUG = False
    TESTING = True

