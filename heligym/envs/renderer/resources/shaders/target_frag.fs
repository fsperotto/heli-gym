#version 330 core
in vec2 TexCoords;
out vec4 FragColor;
void main()
{
    vec3 barycentric = vec3(TexCoords, 1.0 - TexCoords.x - TexCoords.y);
    vec3 blend = smoothstep(vec3(0.0), fwidth(barycentric) * 1.5, barycentric);
    float interior = min(min(blend.x, blend.y), blend.z);
    // Nearly opaque charcoal edges around translucent cyan faces.
    FragColor = mix(vec4(0.03, 0.06, 0.07, 0.95),
                    vec4(0.0, 1.0, 0.85, 0.25), interior);
}
