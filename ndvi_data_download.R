library(MODISTools)
library(tidyverse)
library(terra)
library(lubridate)

dates <- mt_dates(product = "VNP13A1", lat = 50.785931, lon = 7.111506)$calendar_date

bonn_ndvi <- mt_subset(product = "VNP13A1",
                       lat = 50.785931,
                       lon = 7.111506,
                       band = c("500_m_16_days_NDVI"),
                       start = "2012-01-17",
                       end = "2024-05-24",
                       km_lr = 1,
                       km_ab = 1,
                       site_name = "bonn",
                       internal = TRUE,
                       progress = TRUE)

bonn_med_ndvi <- bonn_ndvi %>% 
  filter(band == "500_m_16_days_NDVI") %>%
  group_by(calendar_date) %>%
  summarise(doy = yday(as_date(calendar_date)),
            ndvi_median = median(value * as.numeric(scale))) %>% 
  distinct(doy, .keep_all = TRUE)

bonn_med_ndvi %>%
  ggplot(aes(x = as_date(calendar_date), y = ndvi_median)) +
  theme_bw() +
  geom_line()