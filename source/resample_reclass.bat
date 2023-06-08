@echo OFF
set saga_dir=C:\saga-8.5.1_x64
set subbasin=winooski
set fp_map_dir=C:\Users\klawson1\Documents\CIROH_Floodplains\tmp_probHAND\%subbasin%\*
set unit_dir=C:\Users\klawson1\Documents\CIROH_Floodplains\%subbasin%
cd %saga_dir%
FOR %%f IN (%fp_map_dir%) DO CALL :test_func %%f, %unit_dir%
PAUSE

:test_func
saga_cmd grid_tools 15 -INPUT %~1 -RESULT %~1 -METHOD 2 -RETAB "C:\Users\klawson1\Documents\CIROH_Floodplains\source\lookup.csv"
saga_cmd grid_tools 11 -INPUT %~1 -OUTPUT %~1 -TYPE 0
set fname=%~n1
MOVE %~1 %~2\subbasins\%fname:~4,4%\rasters\valley_bottom.tif
EXIT /B 0