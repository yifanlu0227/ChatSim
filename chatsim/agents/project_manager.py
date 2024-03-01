import openai 
import traceback
import copy
import openai
from termcolor import colored

class ProjectManager:
    def __init__(self, config):
        self.config = config

    def decompose_prompt(self, scene, user_prompt):
        """ decompose the prompt to the corresponding chatsim.agents.
        Input:
            scene : Scene
                scene object.
            user_prompt : str
                language prompt to ChatSim.
        Return:
            tasks : dict
                a dictionary of decomposed tasks.
        """
        q0 = "I have a requirement of editing operations in an autonomous driving scenario, and I need your help to break it down into one or several supportable actions. The scene is large which means many vehicles can be contained. "
                
        q1 = "The supportable five actions include adding vehicles , \
                deleting vehicles , \
                put back deleted vehicles, \
                adjusting added vehicles , \
                viewpoint adjustment."

        q2 = "Please try to retain all the semantics and adjunct words from the original text. Each adding action should only contain one car. " + \
             "Information about adding vehicles (such as their type, positions, driving status, speed, color, etc.) should be directly included within the adding action." 

        q3 = "Split actions should be stored in a JSON dictonary. The key is action id and the value is specific action. They will be executed sequentially, and the broken operations should be independent with each other and do not rely on the detailed scene information."
        
        q4 = "An example: the requirement is 'substitute the red car in the scene', you break it down and return" + \
              "{ 1: 'Delete the red car from the scene', 2: 'Add a new car at the location where the red car was deleted' }."

        q5 = "An example: the requirement is 'delete the farthest car and add a red Audi in the right front', you break it down and return " + \
              "{ 1: 'Delete the farthest car', 2: 'Add a red Audi in the right front' }"

        q6 = "An example: the requirement is 'delete all cars', you break it down and return " + \
             "{ 1: 'Delete all the cars'} "

        q7 = "I may provide very abstract requirements. For such requirements, you should analyze how to comply with the splitting of actions." 

        q8 = "An example (very abstract): the requirement is 'I want several cars driving slowly in the scene', you analyse and return " + \
             "{ 1: 'Add one car driving slowly', 2 : 'Add one car driving slowly', 3 : 'Add one car driving slowly', 4 : 'Add one car driving slowly', 5 : 'Add one car driving slowly', 6 : 'Add one car driving slowly', 7 : 'Add one car driving slowly'} "

        q9 = "Do not return any code or explanation; only a JSON dictionary is required."

        q10 = "Attention: the adjustments for one specific added vehicle should be included in one single output action. If there are multiple adjustments for one already added car, these adjustments must be merged in one action."
                
        q11 = "Attention: Do not appear information about the vehicles in the other broken actions."

        q12 = "The requirement is:" + user_prompt

        prompt_list = [q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12]

        result = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "system", "content": "You are an assistant helping me to break down the operations."}] + \
                     [{"role": "user", "content": q} for q in prompt_list]
        )
        answer = result['choices'][0]['message']['content']
        
        print(f"{colored('[User prompt]', color='magenta', attrs=['bold'])} {user_prompt}\n")
        print(f"{colored('[Project Manager] decomposing tasks', color='magenta', attrs=['bold'])} \
               \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}")

        try:
            start = answer.index("{")
            answer = answer[start:]
            end = answer.rfind("}")
            answer = answer[:end+1]
            tasks = eval(answer)
            print(f"{colored('[Extracted Response>>>]', attrs=['bold'])} {answer} \n")

        except Exception as e:
            print(e)
            traceback.print_exc()
            return "Can not parse the requirement."
        
        return tasks


    def dispatch_task(self, scene, task, tech_agents):
        """ dispatch the tasks to the corresponding chatsim.agents.
        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """

        operation_category = {1:'adding', 2:'deleting', 3:'adjusting the viewpoint', 4: 'putting back previously deleted vehicles', 5:'operating on previously added vehicles'}

        q0 = "I will provide you with an action, and you will help me determine which operation this action belongs to."
        
        q1 = "Operations include (1) adding (2) deleting, (3) adjusting the viewpoint, (4) putting back previously deleted vehicles, (5) operating on previously added vehicles."
        
        q2 = "Return the information in JSON format, with a key named 'operation'."

        q3 = "An Example: Given action 'Remove the red car from the scene', you should return {'operation': 2}"

        q4 = "An Example: Given action 'Add a green Porsche at the location where the red car was removed', you should return {'operation': 1}"

        q5 = "An Example: Given action 'Put back the deleted white car', you should return {'operation': 4}"

        q6 = "An Example: Given action 'Move the car just added to the right by 2m', you should return {'operation': 5}"

        q7 = "Note that you should not return any code or explanations, only provide a JSON dictionary."
        
        q8 = task

        prompt_list = [q0,q1,q2,q3,q4,q5,q6,q7,q8]
        result = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": "You are an assistant helping me to classify operations."}] + \
                             [{"role": "user", "content": q} for q in prompt_list]
                )
                
        answer = result['choices'][0]['message']['content']

        print(f"{colored('[Project Manager] dispatching each task', color='magenta', attrs=['bold'])} \
                \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}")

        start = answer.index("{")
        answer = answer[start:]
        end = answer.rfind("}")
        answer = answer[:end+1]
        operation = eval(answer)['operation']
        print(f"{colored('[Extracted Response>>>]', attrs=['bold'])} {operation}. ({operation_category[operation]}) \n")

        if operation == 1:
            self.addition_operation(scene, task, tech_agents)

        elif operation == 2:
            self.deletion_operation(scene, task, tech_agents)

        elif operation == 3:
            self.view_adjust_operation(scene, task, tech_agents)

        elif operation == 4:
            self.put_back_deleted_operation(scene, task, tech_agents)

        elif operation == 5:
            self.revise_added_operation(scene, task)

        scene.past_operations.append(task)

    def addition_operation(self, scene, task, tech_agents):
        """ addition operation. 
        Participants: asset_select_agent, motion_agent

        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        asset_select_agent = tech_agents['asset_select_agent']
        motion_agent = tech_agents['motion_agent']

        placement_mode = motion_agent.llm_reasoning_dependency(scene, task)

        # scene-independent placement -> LLM will determine 'mode', 'distance_constraint', 'distance_min_max'
        if placement_mode['dependency'] == 0:
            placement_prior = motion_agent.llm_placement_wo_dependency(scene, task)
        # scene-dependent placement -> LLM will determine 'x', 'y'
        else:
            valid_object_descriptors = ['x', 'y', 'u', 'v', 'depth', 'rgb']
            scene_object_description = {}
            for car_name, description_dict in scene.original_cars_dict.items():
                filtered_description_dict = {k: v for k, v in description_dict.items() if k in valid_object_descriptors}
                scene_object_description[car_name] = filtered_description_dict

            placement_prior = motion_agent.llm_placement_w_dependency(scene, task, scene_object_description)

        asset_color_and_type = asset_select_agent.llm_selecting_asset(scene, task)
        motion_prior = motion_agent.llm_motion_planning(scene, task)

        added_car_name = scene.add_car({**asset_color_and_type, **placement_prior, **motion_prior})
        motion_agent.func_placement_and_motion_single_vehicle(scene, added_car_name)

    def deletion_operation(self, scene, task, tech_agents):
        """ deletion operation. 
        Participants: deletion_agent

        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        deletion_agent = tech_agents['deletion_agent']

        valid_object_descriptors = ['u', 'v', 'depth', 'rgb']
        scene_object_description = {}
        for car_name, description_dict in scene.original_cars_dict.items():
            filtered_description_dict = {k: v for k, v in description_dict.items() if k in valid_object_descriptors}
            scene_object_description[car_name] = filtered_description_dict

        deletion_car_names = deletion_agent.llm_finding_deletion(scene, task, scene_object_description)
        for car_name in range(deletion_car_names):
            scene.remove_car(car_name)

    def view_adjust_operation(self, scene, task, tech_agents):
        """ view adjust operation. 
        Participants: view_adjust_agent

        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        view_adjust_agent = tech_agents['view_adjust_agent']
        delta_extrinsic = view_adjust_agent.llm_view_adjust(scene, task)
        view_adjust_agent.func_update_extrinsic(scene, delta_extrinsic)

    def put_back_deleted_operation(self, scene, task, tech_agents):
        """ put back deleted operation. 
        Participants: deletion_agent

        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        deletion_agent = tech_agents['deletion_agent']

        valid_object_descriptors = ['u', 'v', 'depth', 'rgb']
        scene_object_description = {}
        for car_name, description_dict in scene.original_cars_dict.items():
            filtered_description_dict = {k: v for k, v in description_dict.items() if k in valid_object_descriptors}
            scene_object_description[car_name] = filtered_description_dict

        put_back_car_names = deletion_agent.llm_putting_back_deletion(scene, task, scene_object_description)
        
        for car_name in put_back_car_names:
            scene.removed_cars.remove(car_name)

    def revise_added_operation(self, scene, task, tech_agents):
        """ revised added vehicle 
        Participants: asset_select_agent, motion_agent

        Input:
            scene : Scene
                scene object.
            task : str
                a decomposed task, should be assigned to one/more agents
            tech_agents : dict
                a dictionary of technical agents, helping to reason the task
        Return:
            callback_message : str
                if encounter bugs, record them in callback_message to users
        """
        asset_select_agent = tech_agents['asset_select_agent']
        motion_agent = tech_agents['motion_agent']

        # provide every car in scene.added_cars_dict with attributes of 'x' 'y'.
        # 'x', 'y' refer to position of the first frame of motion
        for added_car_name, added_car_info in scene.added_cars_dict.items():
            added_car_info['x'] = added_car_info['motion'][0][0]
            added_car_info['y'] = added_car_info['motion'][0][1]

        # create a brief dictionary for LLM reasoning, including
        # x, y (placement prior), action, speed, direction (motion prior), color, type
        added_cars_short_dict = copy.deepcopy(scene.added_cars_dict)
        for added_car_name, added_car_info in added_cars_short_dict.items():
            added_car_info.pop('motion')
            # only preserve 'x' 'y' as the placement prior, 
            # deprecate all scene-independent attributes
            if 'mode' in added_car_info: 
                added_car_info.pop('mode')
                added_car_info.pop('distance_constraint')
                added_car_info.pop('distance_min_max')
                added_car_info.pop('need_placement_and_motion')

        modified_car_dict = asset_select_agent.llm_revise_added_cars(scene, task, added_cars_short_dict)

        # update scene.added_cars_dict
        for modified_car_name, modified_car_info in modified_car_dict.items():
            scene.added_cars_dict[modified_car_name]['color'] = modified_car_info['color']
            scene.added_cars_dict[modified_car_name]['type'] = modified_car_info['type']

            # If the following attributes are updated, placement and motion need to updated.
            scene.added_cars_dict[modified_car_name]['need_placement_and_motion'] = False
            check_attributes = ['action', 'speed', 'direction', 'x', 'y']
            for attri in check_attributes:
                if scene.added_cars_dict[modified_car_name][attri] != modified_car_info[attri]:
                    scene.added_cars_dict[modified_car_name]['need_placement_and_motion'] = True
                    scene.added_cars_dict[modified_car_name][attri] = modified_car_info[attri]
            motion_agent.fucn_placement_and_motion_single_vehicle(scene, modified_car_name)


