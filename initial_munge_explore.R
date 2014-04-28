boiler.data = read.csv("cleaned_boiler_data.csv")
boiler.data$boiler_year = as.integer(boiler.data$boiler_install_date > 1980)
