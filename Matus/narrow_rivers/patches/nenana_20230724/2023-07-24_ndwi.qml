<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology">
 <pipe>
  <rasterrenderer type="singlebandpseudocolor" band="1"
                  opacity="1" classificationMin="-1"
                  classificationMax="1" nodataColor="">
   <rastershader>
    <colorrampshader colorRampType="INTERPOLATED" clip="0"
                     classificationMode="1" labelPrecision="2"
                     minimumValue="-1" maximumValue="1">
      <item value="-1.0" color="#8b4513" alpha="255" label="-1.0  dry land"/>
      <item value="-0.3" color="#d8b365" alpha="255" label="-0.3"/>
      <item value="0.0" color="#ffffff" alpha="255" label=" 0.00  water edge"/>
      <item value="0.3" color="#67a9cf" alpha="255" label=" 0.3"/>
      <item value="1.0" color="#0066ff" alpha="255" label=" 1.0  open water"/>
    </colorrampshader>
   </rastershader>
  </rasterrenderer>
  <brightnesscontrast brightness="0" contrast="0"/>
 </pipe>
 <blendMode>0</blendMode>
</qgis>
