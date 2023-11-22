
function conductor_fresnel(f0, bsdf) {
return bsdf * (f0 + (1 - f0) * (1 - abs(VdotH))^5)
}

function fresnel_mix(ior, base, layer) {
f0 = ((1-ior)/(1+ior))^2
fr = f0 + (1 - f0)*(1 - abs(VdotH))^5
return mix(base, layer, fr)
}

function specular_brdf(α) {
return V * D
}

function diffuse_brdf(color) {
return (1/pi) * color
}

metal_brdf = specular_brdf(roughness^2) * (baseColor.rgb + (1 - baseColor.rgb) * (1 - abs(VdotH))^5)
          = conductor_fresnel(baseColor.rgb, specular_brdf(roughness^2))

dielectric_brdf = mix(diffuse_brdf(baseColor.rgb), specular_brdf(roughness^2), 0.04 + (1 - 0.04) * (1 - abs(VdotH))^5)
                = fresnel_mix(1.5, diffuse_brdf(baseColor.rgb), specular_brdf(roughness^2))

material = mix(dielectric_brdf, metal_brdf, metallic)
         = dielectric_brdf * (1 - metallic) + metal_brdf * metallic
         = (diffuse_brdf(baseColor.rgb) * (1 - a) + specular_brdf(roughness^2) * a) * (1 - metallic) 
           + specular_brdf(roughness^2) * (baseColor.rgb + (1 - baseColor.rgb) * (1 - abs(VdotH))^5) * metallic 
        = diffuse_brdf(baseColor.rgb) * (1 - (0.04 + (1 - 0.04) * (1 - abs(VdotH))^5)) * (1 - metallic)
           + specular_brdf(roughness^2) * (0.04 + (1 - 0.04) * (1 - abs(VdotH))^5)) * (1 - metallic) 
            + specular_brdf(roughness^2) * (baseColor.rgb + (1 - baseColor.rgb) * (1 - abs(VdotH))^5) * metallic 