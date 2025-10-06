
// The three.js scene: the 3D world where you put objects
const scene = new THREE.Scene();
var is_binary_star=false
const width=window.innerWidth;
const height= window.innerHeight;
var camDist = orbit_radius * 0.7;
// The camera
const camera = new THREE.PerspectiveCamera(
  60,
  width / height,
  1,
  1000000
);

// The renderer: something that draws 3D objects onto the canvas
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(width, height);
renderer.setClearColor(0x000000, 1);
// Append the renderer canvas into <body>
document.body.appendChild(renderer.domElement);

const textureLoader = new THREE.TextureLoader();
const sun_texture= textureLoader.load('sun.webp');
const blue_white = textureLoader.load('blue_white.jpg')
const blue = textureLoader.load('blue_star.png')
const terrestrial_texture= textureLoader.load('terrestrial.jpg')

// A Planet we are going to animate
var Planet = {
  // The geometry: the shape & size of the object
  geometry: new THREE.SphereGeometry(1, 40, 40),
  // The material: the appearance (color, texture) of the object
  material: new THREE.MeshStandardMaterial({ map: terrestrial_texture})
}
Planet.mesh = new THREE.Mesh(Planet.geometry, Planet.material);

// Add the Planet into the scene
Planet.mesh.name="planet"
scene.add(Planet.mesh);

var sun = {
  // The geometry: the shape & size of the object
  geometry: new THREE.SphereGeometry(10, 20, 20),
  // The material: the appearance (color, texture) of the object
  material: new THREE.MeshStandardMaterial({ map: blue })
}
sun.mesh = new THREE.Mesh(sun.geometry, sun.material);

var sun_colors = {//colors and intensities of light for sun [color, intensity]
  blue: [0x3fddff,10,0x3feaff,4], //use blue_star texture for blue
  blue_white: [0x3fddff,20,0x3feaff,8],
  white: [0x3fddff,55,0x3feaff,10], //use other texture for white, blue-white
  yellow_white: [0xdfa010,55,0xdea010,30],
  yellow: [0xffa010,35,0xffff10,1],
  orange: [0xff5030,30,0xffffff,1],
  red: [0xff1010,30,0xff1010,1]
}

// Add the Planet into the scene
sun.mesh.name="sun"
scene.add(sun.mesh);

var corona = {
  geometry: new THREE.SphereGeometry(11, 20,20),
  material: new THREE.MeshStandardMaterial({ map: blue, transparent: true, opacity: 0.4})
}

corona.mesh=new THREE.Mesh(corona.geometry, corona.material)
corona.mesh.name="corona"

var planetary_orbit = {
  geometry: new THREE.RingGeometry(30-0.3,30+0.3,100),
  material: new THREE.MeshBasicMaterial({color: 0xffffff})
}
planetary_orbit.mesh=new THREE.Mesh(planetary_orbit.geometry, planetary_orbit.material)
planetary_orbit.mesh.name="planetary_orbit"
planetary_orbit.mesh.rotation.x=-Math.PI/2-0.000001

const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // Soft ambient light
const sun_light = new THREE.PointLight(0xffffff, 0.1, 100); // Bright point light
const sun_spotlight_1 = new THREE.SpotLight(0x3fddff,20,0,Math.PI/2,0.1,0)
//default color if ffaaff and 20i more deviation=more intesenity
const sun_spotlight_2 = new THREE.SpotLight(0x3feaff,8,10,Math.PI/2,1,0)
// Make the camera further from the Planet so we can see it better
var x = 0, y = 0.1, z = 2
camera.position.set(x, y, z)
camera.lookAt(sun.mesh.position.x,sun.mesh.position.y,sun.mesh.position.z)

let isDragging = false;
const previousMousePosition = {
    x: 0,
    y: 0
};

// How fast the camera moves
const moveSpeed = 0.2; 

renderer.domElement.addEventListener('mousedown', (e) => {
    isDragging = true;
    previousMousePosition.x = e.clientX;
    previousMousePosition.y = e.clientY;
});

renderer.domElement.addEventListener('mousemove', (e) => {
    if (!isDragging) return;

    const deltaX = e.clientX - previousMousePosition.x;
    const deltaY = e.clientY - previousMousePosition.y;

    // Move the camera based on the mouse delta
    // Note: we use translateX and translateY to move relative to the camera's local axes
    //camera.translateX(-deltaX * moveSpeed);
    //camera.translateY(deltaY * moveSpeed);

    // Update the previous mouse position
    //previousMousePosition.x = e.clientX;
    //previousMousePosition.y = e.clientY;
});

renderer.domElement.addEventListener('mouseup', () => {
    isDragging = false;
});

renderer.domElement.addEventListener('mouseleave', () => {
    isDragging = false;
});

let sensitivity = 0.001
function onMouseWheel(event) {

    // Prevent the page from scrolling
    event.preventDefault();
    camera.translateOnAxis(camera.position, event.deltaY * sensitivity)
    // The sign of deltaY indicates the direction of the scroll;
}
document.addEventListener('wheel', onMouseWheel)

function starColorChange(){
  let star_temp= 4000
  if(star_temp>=30000){
    sun_spotlight_1.color.setHex(sun_colors.blue[0])
    sun_spotlight_1.intensity= sun_colors.blue[1]
    sun_spotlight_2.color.setHex(sun_colors.blue[2])
    sun_spotlight_2.intensity= sun_colors.blue[3]
    sun.material.map=blue
  }
  else if (star_temp>=10000){
    sun_spotlight_1.color.setHex(sun_colors.blue_white[0])
    sun_spotlight_1.intensity= sun_colors.blue_white[1]
    sun_spotlight_2.color.setHex(sun_colors.blue_white[2])
    sun_spotlight_2.intensity= sun_colors.blue_white[3]
    sun.material.map=blue
  }
  else if (star_temp>=7500){
    sun_spotlight_1.color.setHex(sun_colors.white[0])
    sun_spotlight_1.intensity= sun_colors.white[1]
    sun_spotlight_2.color.setHex(sun_colors.white[2])
    sun_spotlight_2.intensity= sun_colors.white[3]
    sun.material.map=blue
  }
  else if(star_temp>=6000){
    sun_spotlight_1.color.setHex(sun_colors.yellow_white[0])
    sun_spotlight_1.intensity= sun_colors.yellow_white[1]
    sun_spotlight_2.color.setHex(sun_colors.yellow_white[2])
    sun_spotlight_2.intensity= sun_colors.yellow_white[3]
    sun.material.map=sun_texture
  }
  else if (star_temp>=5200){
    sun_spotlight_1.color.setHex(sun_colors.yellow[0])
    sun_spotlight_1.intensity= sun_colors.yellow[1]
    sun_spotlight_2.color.setHex(sun_colors.yellow[2])
    sun_spotlight_2.intensity= sun_colors.yellow[3]
    sun.material.map=sun_texture
  }
  else if (star_temp>=3700){
    sun_spotlight_1.color.setHex(sun_colors.orange[0])
    sun_spotlight_1.intensity= sun_colors.orange[1]
    sun_spotlight_2.color.setHex(sun_colors.orange[2])
    sun_spotlight_2.intensity= sun_colors.orange[3]
    sun.material.map=sun_texture
  }
  else{
    sun_spotlight_1.color.setHex(sun_colors.red[0])
    sun_spotlight_1.intensity= sun_colors.red[1]
    sun_spotlight_2.color.setHex(sun_colors.red[2])
    sun_spotlight_2.intensity= sun_colors.red[3]
    sun.material.map=sun_texture
  }
}
var timeScale = 1;
var period = 2;
var start=Date.now()
var orbit_radius=60
var orbit_speed=0.0005
var planet_radius=1
var sun_radius=10
var planet_rotation_speed=0.05
var psize = 14
var ssize = 1
var planet_star_size_ratio  = psize/(109*ssize)
planet_radius= (planet_star_size_ratio *sun_radius) / 5
orbit_radius=25
var transDuration = 5;
orbit_speed = (5/transDuration) * orbit_speed
camDist = orbit_radius * 0.7 * z
orbit_radius = (orbit_speed*3600* 24*period) / (Math.PI*2)
console.log(camDist)
//camDist= Math.sqrt(((orbit_radius/25)**2) * (camDist**2 + sun_radius**2) - sun_radius**2)
console.log(camDist)
if(orbit_radius < 25){
  orbit_radius = 25
}
console.log(orbit_radius)
var duration;
function render() {
  // Render the scene and the camera
  renderer.render(scene, camera);

  // Rotate the Planet every frame
  Planet.mesh.rotation.y += 0.005;
  //camera.position.set(x*zoom, y*zoom, z*zoom)
  sun.mesh.rotation.y -= 0.01;
  corona.mesh.rotation.y -= 0.005;
  Planet.mesh.rotation.y -= planet_rotation_speed;
  Planet.mesh.position.x = Math.sin((Date.now()-start) * orbit_speed) * orbit_radius;
  Planet.mesh.position.z = Math.cos((Date.now()-start) * orbit_speed) * orbit_radius;
  let distance = Math.sqrt(Math.pow(camera.position.x,2)+Math.pow(camera.position.y,2)+Math.pow(camera.position.z,2))
  sun_spotlight_1.position.x=camera.position.x*25/distance
  sun_spotlight_1.position.y=camera.position.y*25/distance
  sun_spotlight_1.position.z=camera.position.z*25/distance
  sun_spotlight_2.position.x=camera.position.x*25/distance
  sun_spotlight_2.position.y=camera.position.y*25/distance
  sun_spotlight_2.position.z=camera.position.z*25/distance

  //these two if statements provide starting and ending times the planet crossing the sun from the camera's perspective, not just the diameter of the sun
  // 1 second = 10 days in the simulation based on calculations
  //1 orbit at radius = 25 is 12.5 seconds
  if(Math.abs(Planet.mesh.position.x + planet_radius - ( 0 - sun_radius+(orbit_radius/Math.sqrt((orbit_radius*1.4)**2 + sun_radius**2))*sun_radius)) <= .1){
    duration = Date.now()
  }
  if(Math.abs(Planet.mesh.position.x - planet_radius - (sun_radius-(orbit_radius/Math.sqrt((orbit_radius*1.4)**2 + sun_radius**2))*sun_radius)) <= .1){
    duration = Date.now() - duration
    let time = Date.now()
    console.log("Duration is:")
    console.log(duration/1000)
    console.log(time)

  }
  // subtracting the two markers above almost consistently results ~ 0.5 seconds (500 milliseconds)
  //orbit speed of 0.0005 results in a transit duration of 0.5 seconds consistently
  //scene rendered every 1/60 second (60 fps animation)
  requestAnimationFrame(render);
}

function reset(){
    camera.position.set(x, y, z)
    start=Date.now()
    Planet.geometry = new THREE.SphereGeometry(planet_radius, 20, 20)
    Planet.mesh = new THREE.Mesh(Planet.geometry, Planet.material)
    sun.geometry = new THREE.SphereGeometry(sun_radius, 20, 20)
    sun.mesh = new THREE.Mesh(sun.geometry, sun.material);
    planetary_orbit.geometry = new THREE.RingGeometry(orbit_radius-0.1,orbit_radius+0.1,100)
    planetary_orbit.mesh=new THREE.Mesh(planetary_orbit.geometry, planetary_orbit.material)
    planetary_orbit.mesh.name="planetary_orbit"
    scene.remove(scene.getObjectByName("planet"))
    scene.remove(scene.getObjectByName("sun"))
    scene.remove(scene.getObjectByName("planetary_orbit"))
    scene.remove(scene.getObjectByName("sun"))
    scene.remove(scene.getObjectByName("planet"))
    scene.remove(scene.getObjectByName("corona"))
    scene.remove(scene.getObjectByName("planetary_orbit"))
    scene.remove(scene.getObjectByName("sun_1"))
    scene.remove(scene.getObjectByName("sun_2"))
    scene.remove(scene.getObjectByName("corona_1"))
    scene.remove(scene.getObjectByName("corona_2"))
    scene.remove(scene.getObjectByName("binary_orbit"))
    renderer.clear()
    Planet.mesh.name="planet"
    starColorChange()
    sun.mesh.name="sun"
    scene.add(Planet.mesh);
    scene.add(sun.mesh)
    scene.add(ambientLight);
    scene.add(sun_light)
    planetary_orbit.mesh.rotation.x=-Math.PI/2-0.000001
    scene.add(planetary_orbit.mesh)
    scene.add(sun_spotlight_1)
    sun_spotlight_1.position.set(0,0,25)
    sun_spotlight_1.distance=100
    scene.add(sun_spotlight_2)
    sun_spotlight_2.position.set(0,0,25)
    sun_spotlight_2.distance=100
    camera.position.set(x*orbit_radius*0.7, 0, 0.7*z*orbit_radius)
    scene.add(corona.mesh)
}

var binary_orbit_radius=120
var binary_orbit_speed=0.0005

var sun_1 = {
  // The geometry: the shape & size of the object
  geometry: new THREE.SphereGeometry(6, 40, 40),
  // The material: the appearance (color, texture) of the object
  material: new THREE.MeshStandardMaterial({ map: sun_texture})
}
sun_1.mesh = new THREE.Mesh(sun_1.geometry, sun_1.material);

// Add the Planet into the scene
sun_1.mesh.name="sun_1"

var sun_2 = {
  // The geometry: the shape & size of the object
  geometry: new THREE.SphereGeometry(10, 20, 20),
  // The material: the appearance (color, texture) of the object
  material: new THREE.MeshStandardMaterial({ map: sun_texture })
}
sun_2.mesh = new THREE.Mesh(sun_2.geometry, sun_2.material);

// Add the Planet into the scene
sun_2.mesh.name="sun_2"

var corona_1 = {
  geometry: new THREE.SphereGeometry(6*1.1, 20,20),
  material: new THREE.MeshStandardMaterial({ map: sun_texture, transparent: true, opacity: 0.4})
}

corona_1.mesh=new THREE.Mesh(corona_1.geometry, corona_1.material)
corona_1.mesh.name="corona_1"
var corona_2 = {
  geometry: new THREE.SphereGeometry(10*1.1, 20,20),
  material: new THREE.MeshStandardMaterial({ map: sun_texture, transparent: true, opacity: 0.4})
}

corona_2.mesh=new THREE.Mesh(corona_2.geometry, corona_2.material)
corona_2.mesh.name="corona_2"
var center = {
  geometry: new THREE.SphereGeometry(0.8, 20,20),
  material: new THREE.MeshBasicMaterial({color: 0xffffff})
}
center.mesh=new THREE.Mesh(center.geometry, center.material)
var binary_orbit= {
  geometry: new THREE.RingGeometry(binary_orbit_radius-1,binary_orbit_radius+1,100),
  material: new THREE.MeshBasicMaterial({color: 0xffffff})
}
binary_orbit.mesh=new THREE.Mesh(binary_orbit.geometry, binary_orbit.material)
binary_orbit.mesh.rotation.x=-Math.PI/2-0.000001
binary_orbit.mesh.name="binary_orbit"
//const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // Soft ambient light
//const sun_light = new THREE.PointLight(0xffffff, 0.1, 100); // Bright point light
const sun_1_spotlight_1 = new THREE.SpotLight(0xff5030,30,0,Math.PI/2,0.1,0)
//default color if ffaaff and 20i more deviation=more intesenity
const sun_1_spotlight_2 = new THREE.SpotLight(0xffffff,1,10,Math.PI/2,1,0)
const sun_2_spotlight_1 = new THREE.SpotLight(0xff1010,30,0,Math.PI/2,0.1,0)
//default color if ffaaff and 20i more deviation=more intesenity
const sun_2_spotlight_2 = new THREE.SpotLight(0xff1010,1,10,Math.PI/2,1,0)
// Make the camera further from the Planet so we can see it better



function binary_render() {
  // Render the scene and the camera
  renderer.render(scene, camera);
  //binary_orbit.mesh.rotation.x += 0.001
  // Rotate the Planet every frame
  //camera.position.set(x*zoom, y*zoom, z*zoom)
  sun_1.mesh.rotation.y -= 0.03;
  corona_1.mesh.rotation.y -= 0.015;
  sun_2.mesh.rotation.y -= 0.03;
  corona_2.mesh.rotation.y -= 0.015;
  sun_1.mesh.position.x = ((Math.sin(Math.PI-(Date.now()*binary_orbit_speed)) * 1) + 0)*binary_orbit_radius;
  sun_1.mesh.position.z = ((Math.cos(Math.PI-(Date.now()*binary_orbit_speed)) * 1)+0)*binary_orbit_radius;
  sun_2.mesh.position.x = (Math.sin(-(Date.now()*binary_orbit_speed) * 1) - 0)*binary_orbit_radius;
  sun_2.mesh.position.z = (Math.cos(-(Date.now()*binary_orbit_speed) * 1)+0)*binary_orbit_radius;
  corona_1.mesh.position.x=sun_1.mesh.position.x
  corona_1.mesh.position.z=sun_1.mesh.position.z
  corona_2.mesh.position.x=sun_2.mesh.position.x
  corona_2.mesh.position.z=sun_2.mesh.position.z
  sun_1_spotlight_1.target= sun_1.mesh
  sun_1_spotlight_2.target= sun_1.mesh
  sun_2_spotlight_1.target= sun_2.mesh
  sun_2_spotlight_2.target= sun_2.mesh

  // Make it call the render() function about every 1/60 second
  requestAnimationFrame(binary_render);
}

function binary_star(){
  scene.remove(scene.getObjectByName("sun"))
  scene.remove(scene.getObjectByName("planet"))
  scene.remove(scene.getObjectByName("corona"))
  scene.remove(scene.getObjectByName("planetary_orbit"))
  camera.position.set(x, y, z)
  start=Date.now()
  scene.remove(scene.getObjectByName("sun_1"))
  scene.remove(scene.getObjectByName("sun_2"))
  renderer.clear()
  sun_1.mesh.name="sun_1"
  //starColorChange()
  sun_2.mesh.name="sun_2"
  scene.add(sun_1.mesh);
  scene.add(sun_2.mesh)
  scene.add(ambientLight);
  scene.add(center.mesh)
  scene.add(binary_orbit.mesh)
  //scene.add(sun_light)
  scene.add(sun_1_spotlight_1)
  sun_1_spotlight_1.position.set(0,0,1000)
  sun_1_spotlight_1.distance=100
  scene.add(sun_1_spotlight_2)
  sun_1_spotlight_2.position.set(0,0,1000)
  sun_1_spotlight_2.distance=100
  sun_1_spotlight_1.target= sun_1.mesh
  sun_1_spotlight_2.target= sun_1.mesh
  scene.add(sun_2_spotlight_1)
  sun_2_spotlight_1.position.set(0,0,1000)
  sun_2_spotlight_1.distance=100
  scene.add(sun_2_spotlight_2)
  sun_2_spotlight_2.position.set(0,0,1000)
  sun_2_spotlight_2.distance=100
  sun_2_spotlight_1.target= sun_2.mesh
  sun_2_spotlight_2.target= sun_2.mesh
  camera.position.set(x*binary_orbit_radius, y*binary_orbit_radius, z*binary_orbit_radius)
  scene.add(corona_1.mesh)
  scene.add(corona_2.mesh)
}
document.addEventListener('keydown', (event) => {
    if (event.key === 'r') {
        reset()
    }
    if (event.key === 'b'){
      binary_star()
      binary_render()
      is_binary_star=true
    }
})
reset()
render()