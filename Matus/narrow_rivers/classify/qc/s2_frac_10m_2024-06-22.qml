<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
 <pipe>
  <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" classificationMin="0.0" classificationMax="1.0" nodataColor="">
   <rastershader>
    <colorrampshader colorRampType="INTERPOLATED" clip="0" classificationMode="1" labelPrecision="2" minimumValue="0.0" maximumValue="1.0">
      <item value="0.0" color="#ffffff" alpha="255" label="0.0  no water"/>
      <item value="0.25" color="#c6dbef" alpha="255" label="0.25"/>
      <item value="0.5" color="#6baed6" alpha="255" label="0.50  half water"/>
      <item value="0.75" color="#2171b5" alpha="255" label="0.75"/>
      <item value="1.0" color="#08306b" alpha="255" label="1.0  all water"/>
    </colorrampshader>
   </rastershader>
  </rasterrenderer>
  <brightnesscontrast brightness="0" contrast="0"/>
 </pipe>
 <blendMode>0</blendMode>
</qgis>
