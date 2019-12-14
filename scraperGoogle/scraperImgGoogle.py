from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
# initialise les arguments afin de télécharger les fichiers
# arguments = {"keywords":"Tom cruise, Brad Pitt, Jennifer lopez, Bill Gates, Donald Trump, Jean pierre coffe, Marine lepen, jacques chirac, jean lassalle","limit":500,  "print_urls":True, "chromedriver":"chromedriver"}
arguments = {"keywords":"visage, face","limit":1000,  "print_urls":True, "chromedriver":"chromedriver"} 
# Téléchargement des fichiers
paths = response.download(arguments)
print(paths)
