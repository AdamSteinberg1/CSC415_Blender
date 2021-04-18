
######################################################################
## LIBRARIES
######################################################################
from manta import *
import os.path, shutil, math, sys, gc, multiprocessing, platform, time

withMPBake = False # Bake files asynchronously
withMPSave = False # Save files asynchronously
isWindows = platform.system() != 'Darwin' and platform.system() != 'Linux'
# TODO(sebbas): Use this to simulate Windows multiprocessing (has default mode spawn)
#try:
#    multiprocessing.set_start_method('spawn')
#except:
#    pass

######################################################################
## VARIABLES
######################################################################

mantaMsg('Fluid variables')
dim_s351     = 3
res_s351     = 24
gravity_s351 = vec3(0.000000, 0.000000, -9.810000) # in SI unit (e.g. m/s^2)
gs_s351      = vec3(24, 24, 24)
maxVel_s351  = 0

domainClosed_s351     = True
boundConditions_s351  = ''
boundaryWidth_s351    = 1
deleteInObstacle_s351 = False

using_smoke_s351        = False
using_liquid_s351       = True
using_noise_s351        = False
using_adaptTime_s351    = True
using_obstacle_s351     = False
using_guiding_s351      = False
using_fractions_s351    = False
using_invel_s351        = True
using_outflow_s351      = False
using_sndparts_s351     = True
using_speedvectors_s351 = False
using_diffusion_s351    = False

# Fluid time params
timeScale_s351    = 1.000000
timeTotal_s351    = 0.000000
timePerFrame_s351 = 0.000000

# In Blender fluid.c: frame_length = DT_DEFAULT * (25.0 / fps) * time_scale
# with DT_DEFAULT = 0.1
frameLength_s351 = 0.104167
frameLengthUnscaled_s351 = frameLength_s351 / timeScale_s351
frameLengthRaw_s351 = 0.1 * 25 # dt = 0.1 at 25 fps

dt0_s351          = 0.104167
cflCond_s351      = 4.000000
timestepsMin_s351 = 1
timestepsMax_s351 = 4

# Start and stop for simulation
current_frame_s351 = 1
start_frame_s351   = 1
end_frame_s351     = 250

# Fluid diffusion / viscosity
domainSize_s351 = 8.000000 # longest domain side in meters
kinViscosity_s351 = 0.000001 / (domainSize_s351*domainSize_s351) # kinematic viscosity in m^2/s

# Factors to convert Blender units to Manta units
ratioMetersToRes_s351 = float(domainSize_s351) / float(res_s351) # [meters / cells]
mantaMsg('1 Mantaflow cell is ' + str(ratioMetersToRes_s351) + ' Blender length units long.')

ratioResToBLength_s351 = float(res_s351) / float(domainSize_s351) # [cells / blength] (blength: cm, m, or km, ... )
mantaMsg('1 Blender length unit is ' + str(ratioResToBLength_s351) + ' Mantaflow cells long.')

ratioBTimeToTimestep_s351 = float(1) / float(frameLengthRaw_s351) # the time within 1 blender time unit, see also fluid.c
mantaMsg('1 Blender time unit is ' + str(ratioBTimeToTimestep_s351) + ' Mantaflow time units long.')

ratioFrameToFramelength_s351 = float(1) / float(frameLengthUnscaled_s351 ) # the time within 1 frame
mantaMsg('frame / frameLength is ' + str(ratioFrameToFramelength_s351) + ' Mantaflow time units long.')

scaleAcceleration_s351 = ratioResToBLength_s351 * (ratioBTimeToTimestep_s351**2)# [meters/btime^2] to [cells/timestep^2] (btime: sec, min, or h, ...)
mantaMsg('scaleAcceleration is ' + str(scaleAcceleration_s351))

scaleSpeedFrames_s351 = ratioResToBLength_s351 * ratioFrameToFramelength_s351 # [blength/frame] to [cells/frameLength]
mantaMsg('scaleSpeed is ' + str(scaleSpeedFrames_s351))

gravity_s351 *= scaleAcceleration_s351 # scale from world acceleration to cell based acceleration

# OpenVDB options
vdbCompression_s351 = Compression_Blosc
vdbPrecision_s351 = Precision_Half
vdbClip_s351 = 0.000001

# Cache file names
file_data_s351 = 'fluid_data'
file_noise_s351 = 'fluid_noise'
file_mesh_s351 = 'fluid_mesh'
file_meshvel_s351 = 'fluid_mesh'
file_particles_s351 = 'fluid_particles'
file_guiding_s351 = 'fluid_guiding'
mantaMsg('Liquid variables')
narrowBandWidth_s351         = 3
combineBandWidth_s351        = narrowBandWidth_s351 - 1
adjustedNarrowBandWidth_s351 = 3.000000 # only used in adjustNumber to control band width
particleNumber_s351   = 2
minParticles_s351     = 8
maxParticles_s351     = 16
radiusFactor_s351     = 1.000000
using_mesh_s351       = True
using_final_mesh_s351 = True
using_fractions_s351  = False
using_apic_s351       = True
using_viscosity_s351  = False
fracThreshold_s351    = 0.050000
fracDistance_s351     = 0.500000
flipRatio_s351        = 0.970000
concaveUpper_s351     = 3.500000
concaveLower_s351     = 0.400000
meshRadiusFactor_s351 = 2.000000
smoothenPos_s351      = 1
smoothenNeg_s351      = 1
randomness_s351       = 0.100000
surfaceTension_s351   = 0.000000
maxSysParticles_s351  = 0
viscosityValue_s351   = 0.050000

mantaMsg('Fluid variables mesh')
upres_sm351  = 2
gs_sm351     = vec3(upres_sm351*gs_s351.x, upres_sm351*gs_s351.y, upres_sm351*gs_s351.z)

mantaMsg('Fluid variables particles')
upres_sp351  = 1
gs_sp351     = vec3(upres_sp351*gs_s351.x, upres_sp351*gs_s351.y, upres_sp351*gs_s351.z)

tauMin_wc_sp351 = 2.000000
tauMax_wc_sp351 = 8.000000
tauMin_ta_sp351 = 5.000000
tauMax_ta_sp351 = 20.000000
tauMin_k_sp351 = 1.000000
tauMax_k_sp351 = 5.000000
k_wc_sp351 = 200
k_ta_sp351 = 40
k_b_sp351 = 0.500000
k_d_sp351 = 0.600000
lMin_sp351 = 10.000000
lMax_sp351 = 25.000000
c_s_sp351 = 0.4   # classification constant for snd parts
c_b_sp351 = 0.77  # classification constant for snd parts
pot_radius_sp351 = 2
update_radius_sp351 = 2
using_snd_pushout_sp351 = False

######################################################################
## SOLVERS
######################################################################

mantaMsg('Solver base')
s351 = Solver(name='solver_base351', gridSize=gs_s351, dim=dim_s351)

mantaMsg('Solver mesh')
sm351 = Solver(name='solver_mesh351', gridSize=gs_sm351)

mantaMsg('Solver particles')
sp351 = Solver(name='solver_particles351', gridSize=gs_sp351)

######################################################################
## GRIDS
######################################################################

mantaMsg('Fluid alloc data')
flags_s351       = s351.create(FlagGrid, name='flags')
vel_s351         = s351.create(MACGrid, name='velocity', sparse=True)
velTmp_s351      = s351.create(MACGrid, name='velocity_previous', sparse=True)
x_vel_s351       = s351.create(RealGrid, name='x_vel')
y_vel_s351       = s351.create(RealGrid, name='y_vel')
z_vel_s351       = s351.create(RealGrid, name='z_vel')
pressure_s351    = s351.create(RealGrid, name='pressure')
phiObs_s351      = s351.create(LevelsetGrid, name='phi_obstacle')
phiSIn_s351      = s351.create(LevelsetGrid, name='phiSIn') # helper for static flow objects
phiIn_s351       = s351.create(LevelsetGrid, name='phi_inflow')
phiOut_s351      = s351.create(LevelsetGrid, name='phi_out')
forces_s351      = s351.create(Vec3Grid, name='forces')
x_force_s351     = s351.create(RealGrid, name='x_force')
y_force_s351     = s351.create(RealGrid, name='y_force')
z_force_s351     = s351.create(RealGrid, name='z_force')
obvel_s351       = None

# Set some initial values
phiObs_s351.setConst(9999)
phiSIn_s351.setConst(9999)
phiIn_s351.setConst(9999)
phiOut_s351.setConst(9999)

# Keep track of important objects in dict to load them later on
fluid_data_dict_final_s351  = { 'vel' : vel_s351 }
fluid_data_dict_resume_s351 = { 'phiObs' : phiObs_s351, 'phiIn' : phiIn_s351, 'phiOut' : phiOut_s351, 'flags' : flags_s351, 'velTmp' : velTmp_s351 }

mantaMsg('Liquid alloc')
phiParts_s351   = s351.create(LevelsetGrid, name='phi_particles')
phi_s351        = s351.create(LevelsetGrid, name='phi')
phiTmp_s351     = s351.create(LevelsetGrid, name='phi_previous')
velOld_s351     = s351.create(MACGrid, name='velOld')
velParts_s351   = s351.create(MACGrid, name='velParts')
mapWeights_s351 = s351.create(MACGrid, name='mapWeights')
fractions_s351  = None # allocated dynamically
curvature_s351  = None

pp_s351         = s351.create(BasicParticleSystem, name='particles')
pVel_pp351      = pp_s351.create(PdataVec3, name='particles_velocity')

pCx_pp351       = None
pCy_pp351       = None
pCz_pp351       = None
if using_apic_s351:
    pCx_pp351   = pp_s351.create(PdataVec3)
    pCy_pp351   = pp_s351.create(PdataVec3)
    pCz_pp351   = pp_s351.create(PdataVec3)

# Acceleration data for particle nbs
pindex_s351     = s351.create(ParticleIndexSystem, name='pindex')
gpi_s351        = s351.create(IntGrid, name='gpi')

# Keep track of important objects in dict to load them later on
liquid_data_dict_final_s351 = { 'pVel' : pVel_pp351, 'pp' : pp_s351 }
liquid_data_dict_resume_s351 = { 'phiParts' : phiParts_s351, 'phi' : phi_s351, 'phiTmp' : phiTmp_s351 }

mantaMsg('Liquid alloc mesh')
phiParts_sm351 = sm351.create(LevelsetGrid, name='phiParts_mesh')
phi_sm351      = sm351.create(LevelsetGrid, name='phi_mesh')
pp_sm351       = sm351.create(BasicParticleSystem, name='pp_mesh')
flags_sm351    = sm351.create(FlagGrid, name='flags_mesh')
mesh_sm351     = sm351.create(Mesh, name='fluid_mesh')

if using_speedvectors_s351:
    mVel_mesh351 = mesh_sm351.create(MdataVec3, name='vertex_velocities_mesh')
    vel_sm351    = sm351.create(MACGrid, name='velocity_mesh')

# Acceleration data for particle nbs
pindex_sm351  = sm351.create(ParticleIndexSystem, name='pindex_mesh')
gpi_sm351     = sm351.create(IntGrid, name='gpi_mesh')

# Set some initial values
phiParts_sm351.setConst(9999)
phi_sm351.setConst(9999)

# Keep track of important objects in dict to load them later on
liquid_mesh_dict_s351 = { 'lMesh' : mesh_sm351 }

if using_speedvectors_s351:
    liquid_meshvel_dict_s351 = { 'lVelMesh' : mVel_mesh351 }

ppSnd_sp351         = sp351.create(BasicParticleSystem, name='particles_secondary')
pVelSnd_pp351       = ppSnd_sp351.create(PdataVec3, name='particles_velocity_secondary')
pForceSnd_pp351     = ppSnd_sp351.create(PdataVec3, name='particles_force_secondary')
pLifeSnd_pp351      = ppSnd_sp351.create(PdataReal, name='particles_life_secondary')
vel_sp351           = sp351.create(MACGrid, name='velocity_secondary')
flags_sp351         = sp351.create(FlagGrid, name='flags_secondary')
phi_sp351           = sp351.create(LevelsetGrid, name='phi_secondary')
phiObs_sp351        = sp351.create(LevelsetGrid, name='phiObs_secondary')
phiOut_sp351        = sp351.create(LevelsetGrid, name='phiOut_secondary')
normal_sp351        = sp351.create(VecGrid, name='normal_secondary')
neighborRatio_sp351 = sp351.create(RealGrid, name='neighbor_ratio_secondary')
trappedAir_sp351    = sp351.create(RealGrid, name='trapped_air_secondary')
waveCrest_sp351     = sp351.create(RealGrid, name='wave_crest_secondary')
kineticEnergy_sp351 = sp351.create(RealGrid, name='kinetic_energy_secondary')

# Set some initial values
phi_sp351.setConst(9999)
phiObs_sp351.setConst(9999)
phiOut_sp351.setConst(9999)

# Keep track of important objects in dict to load them later on
liquid_particles_dict_final_s351  = { 'pVelSnd' : pVelSnd_pp351, 'pLifeSnd' : pLifeSnd_pp351, 'ppSnd' : ppSnd_sp351 }
liquid_particles_dict_resume_s351 = { 'trappedAir' : trappedAir_sp351, 'waveCrest' : waveCrest_sp351, 'kineticEnergy' : kineticEnergy_sp351 }

mantaMsg('Allocating initial velocity data')
invelC_s351  = s351.create(VecGrid, name='invelC')
x_invel_s351 = s351.create(RealGrid, name='x_invel')
y_invel_s351 = s351.create(RealGrid, name='y_invel')
z_invel_s351 = s351.create(RealGrid, name='z_invel')

######################################################################
## DOMAIN INIT
######################################################################

# Prepare domain
phi_s351.initFromFlags(flags_s351)
phiIn_s351.initFromFlags(flags_s351)

######################################################################
## ADAPTIVE TIME
######################################################################

mantaMsg('Fluid adaptive time stepping')
s351.frameLength  = frameLength_s351
s351.timestepMin  = s351.frameLength / max(1, timestepsMax_s351)
s351.timestepMax  = s351.frameLength / max(1, timestepsMin_s351)
s351.cfl          = cflCond_s351
s351.timePerFrame = timePerFrame_s351
s351.timestep     = dt0_s351
s351.timeTotal    = timeTotal_s351
#mantaMsg('timestep: ' + str(s351.timestep) + ' // timPerFrame: ' + str(s351.timePerFrame) + ' // frameLength: ' + str(s351.frameLength) + ' // timeTotal: ' + str(s351.timeTotal) )

def fluid_adapt_time_step_351():
    mantaMsg('Fluid adapt time step')
    
    # time params are animatable
    s351.frameLength = frameLength_s351
    s351.cfl         = cflCond_s351
    s351.timestepMin  = s351.frameLength / max(1, timestepsMax_s351)
    s351.timestepMax  = s351.frameLength / max(1, timestepsMin_s351)
    
    # ensure that vel grid is full (remember: adaptive domain can reallocate solver)
    copyRealToVec3(sourceX=x_vel_s351, sourceY=y_vel_s351, sourceZ=z_vel_s351, target=vel_s351)
    maxVel_s351 = vel_s351.getMax() if vel_s351 else 0
    if using_adaptTime_s351:
        mantaMsg('Adapt timestep, maxvel: ' + str(maxVel_s351))
        s351.adaptTimestep(maxVel_s351)

######################################################################
## IMPORT
######################################################################

def fluid_file_import_s351(dict, path, framenr, file_format, file_name=None):
    mantaMsg('Fluid file import, frame: ' + str(framenr))
    try:
        framenr = fluid_cache_get_framenr_formatted_351(framenr)
        # New cache: Try to load the data from a single file
        loadCombined = 0
        if file_name is not None:
            file = os.path.join(path, file_name + '_' + framenr + file_format)
            if os.path.isfile(file):
                if file_format == '.vdb':
                    loadCombined = load(name=file, objects=list(dict.values()), worldSize=domainSize_s351)
                elif file_format == '.bobj.gz' or file_format == '.obj':
                    for name, object in dict.items():
                        if os.path.isfile(file):
                            loadCombined = object.load(file)
        
        # Old cache: Try to load the data from separate files, i.e. per object with the object based load() function
        if not loadCombined:
            for name, object in dict.items():
                file = os.path.join(path, name + '_' + framenr + file_format)
                if os.path.isfile(file):
                    loadCombined = object.load(file)
        
        if not loadCombined:
            mantaMsg('Could not load file ' + str(file))
    
    except Exception as e:
        mantaMsg('Exception in Python fluid file import: ' + str(e))
        pass # Just skip file load errors for now

def fluid_cache_get_framenr_formatted_351(framenr):
    return str(framenr).zfill(4) if framenr >= 0 else str(framenr).zfill(5)

def liquid_load_data_351(path, framenr, file_format, resumable):
    mantaMsg('Liquid load data')
    dict = { **fluid_data_dict_final_s351, **fluid_data_dict_resume_s351, **liquid_data_dict_final_s351, **liquid_data_dict_resume_s351 } if resumable else { **fluid_data_dict_final_s351, **liquid_data_dict_final_s351 }
    fluid_file_import_s351(dict=dict, path=path, framenr=framenr, file_format=file_format, file_name=file_data_s351)
    
    copyVec3ToReal(source=vel_s351, targetX=x_vel_s351, targetY=y_vel_s351, targetZ=z_vel_s351)

def liquid_load_mesh_351(path, framenr, file_format):
    mantaMsg('Liquid load mesh')
    dict = liquid_mesh_dict_s351
    fluid_file_import_s351(dict=dict, path=path, framenr=framenr, file_format=file_format, file_name=file_mesh_s351)

def liquid_load_meshvel_351(path, framenr, file_format):
    mantaMsg('Liquid load meshvel')
    dict = liquid_meshvel_dict_s351
    fluid_file_import_s351(dict=dict, path=path, framenr=framenr, file_format=file_format, file_name=file_meshvel_s351)

def liquid_load_particles_351(path, framenr, file_format, resumable):
    mantaMsg('Liquid load particles')
    dict = { **liquid_particles_dict_final_s351, **liquid_particles_dict_resume_s351 } if resumable else { **liquid_particles_dict_final_s351 }
    fluid_file_import_s351(dict=dict, path=path, framenr=framenr, file_format=file_format, file_name=file_particles_s351)

######################################################################
## PRE/POST STEPS
######################################################################

def fluid_pre_step_351():
    mantaMsg('Fluid pre step')
    
    phiObs_s351.setConst(9999)
    phiOut_s351.setConst(9999)
    
    # Main vel grid is copied in adapt time step function
    
    if using_obstacle_s351:
        # Average out velocities from multiple obstacle objects at one cell
        x_obvel_s351.safeDivide(numObs_s351)
        y_obvel_s351.safeDivide(numObs_s351)
        z_obvel_s351.safeDivide(numObs_s351)
        copyRealToVec3(sourceX=x_obvel_s351, sourceY=y_obvel_s351, sourceZ=z_obvel_s351, target=obvelC_s351)
    
    if using_invel_s351:
        copyRealToVec3(sourceX=x_invel_s351, sourceY=y_invel_s351, sourceZ=z_invel_s351, target=invelC_s351)
    
    if using_guiding_s351:
        weightGuide_s351.multConst(0)
        weightGuide_s351.addConst(alpha_sg351)
        interpolateMACGrid(source=guidevel_sg351, target=velT_s351)
        velT_s351.multConst(vec3(gamma_sg351))
    
    x_force_s351.multConst(scaleSpeedFrames_s351)
    y_force_s351.multConst(scaleSpeedFrames_s351)
    z_force_s351.multConst(scaleSpeedFrames_s351)
    copyRealToVec3(sourceX=x_force_s351, sourceY=y_force_s351, sourceZ=z_force_s351, target=forces_s351)
    
    # If obstacle has velocity, i.e. is a moving obstacle, switch to dynamic preconditioner
    if using_smoke_s351 and using_obstacle_s351 and obvelC_s351.getMax() > 0:
        mantaMsg('Using dynamic preconditioner')
        preconditioner_s351 = PcMGDynamic
    else:
        mantaMsg('Using static preconditioner')
        preconditioner_s351 = PcMGStatic

def fluid_post_step_351():
    mantaMsg('Fluid post step')
    
    # Copy vel grid to reals grids (which Blender internal will in turn use for vel access)
    copyVec3ToReal(source=vel_s351, targetX=x_vel_s351, targetY=y_vel_s351, targetZ=z_vel_s351)
    if using_guiding_s351:
        copyVec3ToReal(source=guidevel_sg351, targetX=x_guidevel_s351, targetY=y_guidevel_s351, targetZ=z_guidevel_s351)

######################################################################
## STEPS
######################################################################

def liquid_adaptive_step_351(framenr):
    mantaMsg('Manta step, frame ' + str(framenr))
    s351.frame = framenr
    
    fluid_pre_step_351()
    
    flags_s351.initDomain(boundaryWidth=1 if using_fractions_s351 else 0, phiWalls=phiObs_s351, outflow=boundConditions_s351)
    
    if using_obstacle_s351:
        mantaMsg('Extrapolating object velocity')
        # ensure velocities inside of obs object, slightly add obvels outside of obs object
        # extrapolate with phiObsIn before joining (static) phiObsSIn grid to prevent flows into static obs
        extrapolateVec3Simple(vel=obvelC_s351, phi=phiObsIn_s351, distance=6, inside=True)
        extrapolateVec3Simple(vel=obvelC_s351, phi=phiObsIn_s351, distance=3, inside=False)
        resampleVec3ToMac(source=obvelC_s351, target=obvel_s351)
        
        mantaMsg('Initializing obstacle levelset')
        phiObsIn_s351.join(phiObsSIn_s351) # Join static obstacle map
        phiObsIn_s351.floodFill(boundaryWidth=1)
        extrapolateLsSimple(phi=phiObsIn_s351, distance=6, inside=True)
        extrapolateLsSimple(phi=phiObsIn_s351, distance=3, inside=False)
        phiObs_s351.join(phiObsIn_s351)
        
        # Additional sanity check: fill holes in phiObs which can result after joining with phiObsIn
        phiObs_s351.floodFill(boundaryWidth=2 if using_fractions_s351 else 1)
        extrapolateLsSimple(phi=phiObs_s351, distance=6, inside=True)
        extrapolateLsSimple(phi=phiObs_s351, distance=3)
    
    mantaMsg('Initializing fluid levelset')
    phiIn_s351.join(phiSIn_s351) # Join static flow map
    extrapolateLsSimple(phi=phiIn_s351, distance=6, inside=True)
    extrapolateLsSimple(phi=phiIn_s351, distance=3)
    phi_s351.join(phiIn_s351)
    
    if using_outflow_s351:
        phiOutIn_s351.join(phiOutSIn_s351) # Join static outflow map
        phiOut_s351.join(phiOutIn_s351)
    
    if using_fractions_s351:
        updateFractions(flags=flags_s351, phiObs=phiObs_s351, fractions=fractions_s351, boundaryWidth=boundaryWidth_s351, fracThreshold=fracThreshold_s351)
    setObstacleFlags(flags=flags_s351, phiObs=phiObs_s351, phiOut=phiOut_s351, fractions=fractions_s351, phiIn=phiIn_s351)
    
    if using_obstacle_s351:
        # TODO(sebbas): Enable flags check again, currently produces unstable particle behavior
        phi_s351.subtract(o=phiObsIn_s351) #, flags=flags_s351, subtractType=FlagObstacle)
    
    # add initial velocity: set invel as source grid to ensure const vels in inflow region, sampling makes use of this
    if using_invel_s351:
        extrapolateVec3Simple(vel=invelC_s351, phi=phiIn_s351, distance=6, inside=True)
        # Using cell centered invels, a false isMAC flag ensures correct interpolation
        pVel_pp351.setSource(grid=invelC_s351, isMAC=False)
    # reset pvel grid source before sampling new particles - ensures that new particles are initialized with 0 velocity
    else:
        pVel_pp351.setSource(grid=None, isMAC=False)
    
    pp_s351.maxParticles = maxSysParticles_s351 # remember, 0 means no particle cap
    sampleLevelsetWithParticles(phi=phiIn_s351, flags=flags_s351, parts=pp_s351, discretization=particleNumber_s351, randomness=randomness_s351)
    flags_s351.updateFromLevelset(phi_s351)
    
    mantaMsg('Liquid step / s351.frame: ' + str(s351.frame))
    liquid_step_351()
    
    s351.step()
    
    fluid_post_step_351()

def liquid_step_351():
    mantaMsg('Liquid step')
    
    mantaMsg('Advecting particles')
    pp_s351.advectInGrid(flags=flags_s351, vel=vel_s351, integrationMode=IntRK4, deleteInObstacle=deleteInObstacle_s351, stopInObstacle=False, skipNew=True)
    
    mantaMsg('Pushing particles out of obstacles')
    if using_obstacle_s351 and using_fractions_s351 and fracDistance_s351 > 0:
        # Optional: Increase distance between fluid and obstacles (only obstacles, not borders)
        pushOutofObs(parts=pp_s351, flags=flags_s351, phiObs=phiObsIn_s351, thresh=fracDistance_s351)
    pushOutofObs(parts=pp_s351, flags=flags_s351, phiObs=phiObs_s351)
    
    # save original states for later (used during mesh / secondary particle creation)
    # but only save the state at the beginning of an adaptive frame
    if not s351.timePerFrame:
        phiTmp_s351.copyFrom(phi_s351)
        velTmp_s351.copyFrom(vel_s351)
    
    mantaMsg('Advecting phi')
    advectSemiLagrange(flags=flags_s351, vel=vel_s351, grid=phi_s351, order=1) # first order is usually enough
    mantaMsg('Advecting velocity')
    advectSemiLagrange(flags=flags_s351, vel=vel_s351, grid=vel_s351, order=2)
    
    # create level set of particles
    gridParticleIndex(parts=pp_s351, flags=flags_s351, indexSys=pindex_s351, index=gpi_s351)
    unionParticleLevelset(parts=pp_s351, indexSys=pindex_s351, flags=flags_s351, index=gpi_s351, phi=phiParts_s351, radiusFactor=radiusFactor_s351)
    
    # combine level set of particles with grid level set
    phi_s351.addConst(1.) # shrink slightly
    phi_s351.join(phiParts_s351)
    extrapolateLsSimple(phi=phi_s351, distance=narrowBandWidth_s351+2, inside=True)
    extrapolateLsSimple(phi=phi_s351, distance=3)
    phi_s351.setBoundNeumann(0) # make sure no particles are placed at outer boundary
    
    if not domainClosed_s351 or using_outflow_s351:
        resetOutflow(flags=flags_s351, phi=phi_s351, parts=pp_s351, index=gpi_s351, indexSys=pindex_s351)
    flags_s351.updateFromLevelset(phi_s351)
    
    # combine particle velocities with advected grid velocities
    if using_apic_s351:
        apicMapPartsToMAC(flags=flags_s351, vel=vel_s351, parts=pp_s351, partVel=pVel_pp351, cpx=pCx_pp351, cpy=pCy_pp351, cpz=pCz_pp351)
    else:
        mapPartsToMAC(vel=velParts_s351, flags=flags_s351, velOld=velOld_s351, parts=pp_s351, partVel=pVel_pp351, weight=mapWeights_s351)
    
    extrapolateMACFromWeight(vel=velParts_s351, distance=2, weight=mapWeights_s351)
    combineGridVel(vel=velParts_s351, weight=mapWeights_s351, combineVel=vel_s351, phi=phi_s351, narrowBand=combineBandWidth_s351, thresh=0)
    velOld_s351.copyFrom(vel_s351)
    
    # forces & pressure solve
    addGravity(flags=flags_s351, vel=vel_s351, gravity=gravity_s351, scale=False)
    
    mantaMsg('Adding external forces')
    addForceField(flags=flags_s351, vel=vel_s351, force=forces_s351)
    
    extrapolateMACSimple(flags=flags_s351, vel=vel_s351, distance=2, intoObs=True if using_fractions_s351 else False)
    
    # vel diffusion / viscosity!
    if using_diffusion_s351:
        mantaMsg('Viscosity')
        # diffusion param for solve = const * dt / dx^2
        alphaV = kinViscosity_s351 * s351.timestep * float(res_s351*res_s351)
        setWallBcs(flags=flags_s351, vel=vel_s351, obvel=None if using_fractions_s351 else obvel_s351, phiObs=phiObs_s351, fractions=fractions_s351)
        cgSolveDiffusion(flags_s351, vel_s351, alphaV)
        
        mantaMsg('Curvature')
        getLaplacian(laplacian=curvature_s351, grid=phi_s351)
        curvature_s351.clamp(-1.0, 1.0)
    
    setWallBcs(flags=flags_s351, vel=vel_s351, obvel=None if using_fractions_s351 else obvel_s351, phiObs=phiObs_s351, fractions=fractions_s351)
    if using_viscosity_s351:
        viscosity_s351.setConst(viscosityValue_s351)
        applyViscosity(flags=flags_s351, phi=phi_s351, vel=vel_s351, volumes=volumes_s351, viscosity=viscosity_s351)
    
    setWallBcs(flags=flags_s351, vel=vel_s351, obvel=None if using_fractions_s351 else obvel_s351, phiObs=phiObs_s351, fractions=fractions_s351)
    if using_guiding_s351:
        mantaMsg('Guiding and pressure')
        PD_fluid_guiding(vel=vel_s351, velT=velT_s351, flags=flags_s351, phi=phi_s351, curv=curvature_s351, surfTens=surfaceTension_s351, fractions=fractions_s351, weight=weightGuide_s351, blurRadius=beta_sg351, pressure=pressure_s351, tau=tau_sg351, sigma=sigma_sg351, theta=theta_sg351, zeroPressureFixing=domainClosed_s351)
    else:
        mantaMsg('Pressure')
        solvePressure(flags=flags_s351, vel=vel_s351, pressure=pressure_s351, curv=curvature_s351, surfTens=surfaceTension_s351, fractions=fractions_s351, obvel=obvel_s351 if using_fractions_s351 else None, zeroPressureFixing=domainClosed_s351)
    
    extrapolateMACSimple(flags=flags_s351, vel=vel_s351, distance=4, intoObs=True if using_fractions_s351 else False)
    setWallBcs(flags=flags_s351, vel=vel_s351, obvel=None if using_fractions_s351 else obvel_s351, phiObs=phiObs_s351, fractions=fractions_s351)
    
    if not using_fractions_s351:
        extrapolateMACSimple(flags=flags_s351, vel=vel_s351)
    
    # set source grids for resampling, used in adjustNumber!
    pVel_pp351.setSource(grid=vel_s351, isMAC=True)
    adjustNumber(parts=pp_s351, vel=vel_s351, flags=flags_s351, minParticles=minParticles_s351, maxParticles=maxParticles_s351, phi=phi_s351, exclude=phiObs_s351, radiusFactor=radiusFactor_s351, narrowBand=adjustedNarrowBandWidth_s351)
    
    if using_apic_s351:
        apicMapMACGridToParts(partVel=pVel_pp351, cpx=pCx_pp351, cpy=pCy_pp351, cpz=pCz_pp351, parts=pp_s351, vel=vel_s351, flags=flags_s351)
    else:
        flipVelocityUpdate(vel=vel_s351, velOld=velOld_s351, flags=flags_s351, parts=pp_s351, partVel=pVel_pp351, flipRatio=flipRatio_s351)

def liquid_step_mesh_351():
    mantaMsg('Liquid step mesh')
    
    # no upres: just use the loaded grids
    if upres_sm351 <= 1:
        phi_sm351.copyFrom(phi_s351)
    
    # with upres: recreate grids
    else:
        interpolateGrid(target=phi_sm351, source=phi_s351)
    
    # create surface
    pp_sm351.readParticles(pp_s351)
    gridParticleIndex(parts=pp_sm351, flags=flags_sm351, indexSys=pindex_sm351, index=gpi_sm351)
    
    if using_final_mesh_s351:
        mantaMsg('Liquid using improved particle levelset')
        improvedParticleLevelset(pp_sm351, pindex_sm351, flags_sm351, gpi_sm351, phiParts_sm351, meshRadiusFactor_s351, smoothenPos_s351, smoothenNeg_s351, concaveLower_s351, concaveUpper_s351)
    else:
        mantaMsg('Liquid using union particle levelset')
        unionParticleLevelset(pp_sm351, pindex_sm351, flags_sm351, gpi_sm351, phiParts_sm351, meshRadiusFactor_s351)
    
    phi_sm351.addConst(1.) # shrink slightly
    phi_sm351.join(phiParts_sm351)
    extrapolateLsSimple(phi=phi_sm351, distance=narrowBandWidth_s351+2, inside=True)
    extrapolateLsSimple(phi=phi_sm351, distance=3)
    phi_sm351.setBoundNeumann(0) # make sure no particles are placed at outer boundary
    
    # Vert vel vector needs to pull data from vel grid with correct dim
    if using_speedvectors_s351:
        interpolateMACGrid(target=vel_sm351, source=vel_s351)
        mVel_mesh351.setSource(grid=vel_sm351, isMAC=True)
    
    # Set 0.5 boundary at walls + account for extra wall thickness in fractions mode + account for grid scaling:
    # E.g. at upres=1 we expect 1 cell border (or 2 with fractions), at upres=2 we expect 2 cell border (or 4 with fractions), etc.
    # Use -1 since setBound() starts counting at 0 (and additional -1 for fractions to account for solid/fluid interface cells)
    phi_sm351.setBound(value=0.5, boundaryWidth=(upres_sm351*2)-2 if using_fractions_s351 else upres_sm351-1)
    phi_sm351.createMesh(mesh_sm351)

def liquid_step_particles_351():
    mantaMsg('Secondary particles step')
    
    # no upres: just use the loaded grids
    if upres_sp351 <= 1:
        vel_sp351.copyFrom(velTmp_s351)
        phiObs_sp351.copyFrom(phiObs_s351)
        phi_sp351.copyFrom(phiTmp_s351)
        phiOut_sp351.copyFrom(phiOut_s351)
    
    # with upres: recreate grids
    else:
        # create highres grids by interpolation
        interpolateMACGrid(target=vel_sp351, source=velTmp_s351)
        interpolateGrid(target=phiObs_sp351, source=phiObs_s351)
        interpolateGrid(target=phi_sp351, source=phiTmp_s351)
        interpolateGrid(target=phiOut_sp351, source=phiOut_s351)
    
    # phiIn not needed, bwidth to 0 because we are omitting flags.initDomain()
    setObstacleFlags(flags=flags_sp351, phiObs=phiObs_sp351, phiOut=phiOut_sp351, phiIn=None, boundaryWidth=0)
    flags_sp351.updateFromLevelset(levelset=phi_sp351)
    
    # Actual secondary particle simulation
    flipComputeSecondaryParticlePotentials(potTA=trappedAir_sp351, potWC=waveCrest_sp351, potKE=kineticEnergy_sp351, neighborRatio=neighborRatio_sp351, flags=flags_sp351, v=vel_sp351, normal=normal_sp351, phi=phi_sp351, radius=pot_radius_sp351, tauMinTA=tauMin_ta_sp351, tauMaxTA=tauMax_ta_sp351, tauMinWC=tauMin_wc_sp351, tauMaxWC=tauMax_wc_sp351, tauMinKE=tauMin_k_sp351, tauMaxKE=tauMax_k_sp351, scaleFromManta=ratioMetersToRes_s351)
    flipSampleSecondaryParticles(mode='single', flags=flags_sp351, v=vel_sp351, pts_sec=ppSnd_sp351, v_sec=pVelSnd_pp351, l_sec=pLifeSnd_pp351, lMin=lMin_sp351, lMax=lMax_sp351, potTA=trappedAir_sp351, potWC=waveCrest_sp351, potKE=kineticEnergy_sp351, neighborRatio=neighborRatio_sp351, c_s=c_s_sp351, c_b=c_b_sp351, k_ta=k_ta_sp351, k_wc=k_wc_sp351)
    flipUpdateSecondaryParticles(mode='linear', pts_sec=ppSnd_sp351, v_sec=pVelSnd_pp351, l_sec=pLifeSnd_pp351, f_sec=pForceSnd_pp351, flags=flags_sp351, v=vel_sp351, neighborRatio=neighborRatio_sp351, radius=update_radius_sp351, gravity=gravity_s351, scale=False, k_b=k_b_sp351, k_d=k_d_sp351, c_s=c_s_sp351, c_b=c_b_sp351)
    if using_snd_pushout_sp351:
        pushOutofObs(parts=ppSnd_sp351, flags=flags_sp351, phiObs=phiObs_sp351, shift=1.0)
    flipDeleteParticlesInObstacle(pts=ppSnd_sp351, flags=flags_sp351) # delete particles inside obstacle and outflow cells
    
    # Print debug information in the console
    if 0:
        debugGridInfo(flags=flags_sp351, grid=trappedAir_sp351, name='Trapped Air')
        debugGridInfo(flags=flags_sp351, grid=waveCrest_sp351, name='Wave Crest')
        debugGridInfo(flags=flags_sp351, grid=kineticEnergy_sp351, name='Kinetic Energy')

######################################################################
## MAIN
######################################################################

# Helper function to call cache load functions
def load_data(frame, cache_resumable):
    liquid_load_data_351(os.path.join(cache_dir, 'data'), frame, file_format_data, cache_resumable)
    if using_sndparts_s351:
        liquid_load_particles_351(os.path.join(cache_dir, 'particles'), frame, file_format_data, cache_resumable)
    if using_mesh_s351:
        liquid_load_mesh_351(os.path.join(cache_dir, 'mesh'), frame, file_format_mesh)
    if using_guiding_s351:
        fluid_load_guiding_351(os.path.join(cache_dir, 'guiding'), frame, file_format_data)

# Helper function to call step functions
def step(frame):
    liquid_adaptive_step_351(frame)
    if using_mesh_s351:
        liquid_step_mesh_351()
    if using_sndparts_s351:
        liquid_step_particles_351()

gui = None
if (GUI):
    gui=Gui()
    gui.show()
    gui.pause()

cache_resumable       = False
cache_dir             = 'D:\blender-fluid-sim\cache'
file_format_data      = '.vdb'
file_format_mesh      = '.bobj.gz'

# How many frame to load from cache
from_cache_cnt = 100

loop_cnt = 0
while current_frame_s351 <= end_frame_s351:
    
    # Load already simulated data from cache:
    if loop_cnt < from_cache_cnt:
        load_data(current_frame_s351, cache_resumable)
    
    # Otherwise simulate new data
    else:
        while(s351.frame <= current_frame_s351):
            if using_adaptTime_s351:
                fluid_adapt_time_step_351()
            step(current_frame_s351)
    
    current_frame_s351 += 1
    loop_cnt += 1
    
    if gui:
        gui.pause()
