#By Jaikuon
#My first ML project. This was a attempt to predict the origin of a name using ML.



import model
import flaglist 

def passport(name):
    prediction = model.usemodel(name)
    return prediction

def main():
    name_input = str(input("Input Name: "))
    country = passport(name_input)
    country=country[0]
    if country in flaglist.flags:
        country = flaglist.flags[country]
        print(f"Are you from {country}?")
    else:
        print("Country not Found")
        main()

if __name__ == "__main__":
    main()
    
