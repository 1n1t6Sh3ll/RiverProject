<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology">
 <renderer-v2 type="categorizedSymbol" attr="audit_kind"
              forceraster="0" symbollevels="0" enableorderby="0">
  <categories>
   <category render="true" value="river_dry" symbol="0" label="river_dry - labelled river, no water found"/>
   <category render="true" value="land_wet" symbol="1" label="land_wet - labelled land, mostly water"/>
   <category render="true" value="shadow_suspect" symbol="2" label="shadow_suspect - dark in NIR and green: shadow, not water"/>
   <category render="true" value="ndwi_denies" symbol="3" label="ndwi_denies - unmix says water, NDWI says not"/>
   <category render="true" value="disagree" symbol="4" label="disagree - the estimates disagree"/>
  </categories>
  <symbols>
  <symbol type="marker" name="0" alpha="1" clip_to_extent="1" force_rhr="0">
   <layer class="SimpleMarker" enabled="1" pass="0" locked="0">
    <Option type="Map">
        <Option name="name" type="QString" value="circle"/>
        <Option name="color" type="QString" value="227,26,28,255"/>
        <Option name="outline_color" type="QString" value="0,0,0,255"/>
        <Option name="outline_style" type="QString" value="solid"/>
        <Option name="outline_width" type="QString" value="0.2"/>
        <Option name="outline_width_unit" type="QString" value="MM"/>
        <Option name="size" type="QString" value="2.2"/>
        <Option name="size_unit" type="QString" value="MM"/>
        <Option name="angle" type="QString" value="0"/>
        <Option name="offset" type="QString" value="0,0"/>
        <Option name="offset_unit" type="QString" value="MM"/>
        <Option name="scale_method" type="QString" value="diameter"/>
        <Option name="joinstyle" type="QString" value="bevel"/>
        <Option name="cap_style" type="QString" value="square"/>
        <Option name="horizontal_anchor_point" type="QString" value="1"/>
        <Option name="vertical_anchor_point" type="QString" value="1"/>
    </Option>
      <prop k="name" v="circle"/>
      <prop k="color" v="227,26,28,255"/>
      <prop k="outline_color" v="0,0,0,255"/>
      <prop k="outline_style" v="solid"/>
      <prop k="outline_width" v="0.2"/>
      <prop k="outline_width_unit" v="MM"/>
      <prop k="size" v="2.2"/>
      <prop k="size_unit" v="MM"/>
      <prop k="angle" v="0"/>
      <prop k="offset" v="0,0"/>
      <prop k="offset_unit" v="MM"/>
      <prop k="scale_method" v="diameter"/>
      <prop k="joinstyle" v="bevel"/>
      <prop k="cap_style" v="square"/>
      <prop k="horizontal_anchor_point" v="1"/>
      <prop k="vertical_anchor_point" v="1"/>
   </layer>
  </symbol>
  <symbol type="marker" name="1" alpha="1" clip_to_extent="1" force_rhr="0">
   <layer class="SimpleMarker" enabled="1" pass="0" locked="0">
    <Option type="Map">
        <Option name="name" type="QString" value="triangle"/>
        <Option name="color" type="QString" value="255,191,0,255"/>
        <Option name="outline_color" type="QString" value="0,0,0,255"/>
        <Option name="outline_style" type="QString" value="solid"/>
        <Option name="outline_width" type="QString" value="0.2"/>
        <Option name="outline_width_unit" type="QString" value="MM"/>
        <Option name="size" type="QString" value="2.2"/>
        <Option name="size_unit" type="QString" value="MM"/>
        <Option name="angle" type="QString" value="0"/>
        <Option name="offset" type="QString" value="0,0"/>
        <Option name="offset_unit" type="QString" value="MM"/>
        <Option name="scale_method" type="QString" value="diameter"/>
        <Option name="joinstyle" type="QString" value="bevel"/>
        <Option name="cap_style" type="QString" value="square"/>
        <Option name="horizontal_anchor_point" type="QString" value="1"/>
        <Option name="vertical_anchor_point" type="QString" value="1"/>
    </Option>
      <prop k="name" v="triangle"/>
      <prop k="color" v="255,191,0,255"/>
      <prop k="outline_color" v="0,0,0,255"/>
      <prop k="outline_style" v="solid"/>
      <prop k="outline_width" v="0.2"/>
      <prop k="outline_width_unit" v="MM"/>
      <prop k="size" v="2.2"/>
      <prop k="size_unit" v="MM"/>
      <prop k="angle" v="0"/>
      <prop k="offset" v="0,0"/>
      <prop k="offset_unit" v="MM"/>
      <prop k="scale_method" v="diameter"/>
      <prop k="joinstyle" v="bevel"/>
      <prop k="cap_style" v="square"/>
      <prop k="horizontal_anchor_point" v="1"/>
      <prop k="vertical_anchor_point" v="1"/>
   </layer>
  </symbol>
  <symbol type="marker" name="2" alpha="1" clip_to_extent="1" force_rhr="0">
   <layer class="SimpleMarker" enabled="1" pass="0" locked="0">
    <Option type="Map">
        <Option name="name" type="QString" value="square"/>
        <Option name="color" type="QString" value="117,112,179,255"/>
        <Option name="outline_color" type="QString" value="0,0,0,255"/>
        <Option name="outline_style" type="QString" value="solid"/>
        <Option name="outline_width" type="QString" value="0.2"/>
        <Option name="outline_width_unit" type="QString" value="MM"/>
        <Option name="size" type="QString" value="2.2"/>
        <Option name="size_unit" type="QString" value="MM"/>
        <Option name="angle" type="QString" value="0"/>
        <Option name="offset" type="QString" value="0,0"/>
        <Option name="offset_unit" type="QString" value="MM"/>
        <Option name="scale_method" type="QString" value="diameter"/>
        <Option name="joinstyle" type="QString" value="bevel"/>
        <Option name="cap_style" type="QString" value="square"/>
        <Option name="horizontal_anchor_point" type="QString" value="1"/>
        <Option name="vertical_anchor_point" type="QString" value="1"/>
    </Option>
      <prop k="name" v="square"/>
      <prop k="color" v="117,112,179,255"/>
      <prop k="outline_color" v="0,0,0,255"/>
      <prop k="outline_style" v="solid"/>
      <prop k="outline_width" v="0.2"/>
      <prop k="outline_width_unit" v="MM"/>
      <prop k="size" v="2.2"/>
      <prop k="size_unit" v="MM"/>
      <prop k="angle" v="0"/>
      <prop k="offset" v="0,0"/>
      <prop k="offset_unit" v="MM"/>
      <prop k="scale_method" v="diameter"/>
      <prop k="joinstyle" v="bevel"/>
      <prop k="cap_style" v="square"/>
      <prop k="horizontal_anchor_point" v="1"/>
      <prop k="vertical_anchor_point" v="1"/>
   </layer>
  </symbol>
  <symbol type="marker" name="3" alpha="1" clip_to_extent="1" force_rhr="0">
   <layer class="SimpleMarker" enabled="1" pass="0" locked="0">
    <Option type="Map">
        <Option name="name" type="QString" value="pentagon"/>
        <Option name="color" type="QString" value="106,61,154,255"/>
        <Option name="outline_color" type="QString" value="0,0,0,255"/>
        <Option name="outline_style" type="QString" value="solid"/>
        <Option name="outline_width" type="QString" value="0.2"/>
        <Option name="outline_width_unit" type="QString" value="MM"/>
        <Option name="size" type="QString" value="2.2"/>
        <Option name="size_unit" type="QString" value="MM"/>
        <Option name="angle" type="QString" value="0"/>
        <Option name="offset" type="QString" value="0,0"/>
        <Option name="offset_unit" type="QString" value="MM"/>
        <Option name="scale_method" type="QString" value="diameter"/>
        <Option name="joinstyle" type="QString" value="bevel"/>
        <Option name="cap_style" type="QString" value="square"/>
        <Option name="horizontal_anchor_point" type="QString" value="1"/>
        <Option name="vertical_anchor_point" type="QString" value="1"/>
    </Option>
      <prop k="name" v="pentagon"/>
      <prop k="color" v="106,61,154,255"/>
      <prop k="outline_color" v="0,0,0,255"/>
      <prop k="outline_style" v="solid"/>
      <prop k="outline_width" v="0.2"/>
      <prop k="outline_width_unit" v="MM"/>
      <prop k="size" v="2.2"/>
      <prop k="size_unit" v="MM"/>
      <prop k="angle" v="0"/>
      <prop k="offset" v="0,0"/>
      <prop k="offset_unit" v="MM"/>
      <prop k="scale_method" v="diameter"/>
      <prop k="joinstyle" v="bevel"/>
      <prop k="cap_style" v="square"/>
      <prop k="horizontal_anchor_point" v="1"/>
      <prop k="vertical_anchor_point" v="1"/>
   </layer>
  </symbol>
  <symbol type="marker" name="4" alpha="1" clip_to_extent="1" force_rhr="0">
   <layer class="SimpleMarker" enabled="1" pass="0" locked="0">
    <Option type="Map">
        <Option name="name" type="QString" value="diamond"/>
        <Option name="color" type="QString" value="27,158,119,255"/>
        <Option name="outline_color" type="QString" value="0,0,0,255"/>
        <Option name="outline_style" type="QString" value="solid"/>
        <Option name="outline_width" type="QString" value="0.2"/>
        <Option name="outline_width_unit" type="QString" value="MM"/>
        <Option name="size" type="QString" value="2.2"/>
        <Option name="size_unit" type="QString" value="MM"/>
        <Option name="angle" type="QString" value="0"/>
        <Option name="offset" type="QString" value="0,0"/>
        <Option name="offset_unit" type="QString" value="MM"/>
        <Option name="scale_method" type="QString" value="diameter"/>
        <Option name="joinstyle" type="QString" value="bevel"/>
        <Option name="cap_style" type="QString" value="square"/>
        <Option name="horizontal_anchor_point" type="QString" value="1"/>
        <Option name="vertical_anchor_point" type="QString" value="1"/>
    </Option>
      <prop k="name" v="diamond"/>
      <prop k="color" v="27,158,119,255"/>
      <prop k="outline_color" v="0,0,0,255"/>
      <prop k="outline_style" v="solid"/>
      <prop k="outline_width" v="0.2"/>
      <prop k="outline_width_unit" v="MM"/>
      <prop k="size" v="2.2"/>
      <prop k="size_unit" v="MM"/>
      <prop k="angle" v="0"/>
      <prop k="offset" v="0,0"/>
      <prop k="offset_unit" v="MM"/>
      <prop k="scale_method" v="diameter"/>
      <prop k="joinstyle" v="bevel"/>
      <prop k="cap_style" v="square"/>
      <prop k="horizontal_anchor_point" v="1"/>
      <prop k="vertical_anchor_point" v="1"/>
   </layer>
  </symbol>
  </symbols>
 </renderer-v2>
</qgis>
