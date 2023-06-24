
from typing import Any, List, Callable, Tuple
import itertools
from collections import Counter
import json



def nested_map(operation: Callable[[Any], Any], note_array: Any) -> List[Any]:
    if isinstance(note_array, list):
        return [nested_map(operation, x) for x in note_array]
    else:
        return operation(note_array)
    
def nested_zip(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return [nested_zip(sub_a, sub_b) for sub_a, sub_b in zip(a, b)]
    else:
        return [a, b]  
    

def map_range(value, from_min, from_max, to_min, to_max):
    """
    Maps a value from one range to another range.
    """
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled_value = float(value - from_min) / float(from_range)
    return to_min + (scaled_value * to_range)    

def is_odd(num):
    return num % 2 != 0

class Note:
    def __init__():
        return 

class Notation(Note):
    def __init__(self, pitch: int, value :int = 1):
        self.pitch = pitch
        self.value = value

class Empty(Notation):
    def __init__(self):
        self.pitch = None
        self.value = None


class Note_Out:
    def __init__(self, pitch, start_time, end_time):
        self.pitch = pitch
        self.start_time = start_time
        self.end_time = end_time

    def __dict__(self):
        return {
            "pitch": self.pitch,
            "start_time": self.start_time,
            "end_time": self.end_time
        }   


class Cluster:
    # Cluster should interference with the original sequence
    # e.g, zoom, group, progression, chordify
    def __init__(self, note_array: List[Any]):
        self.note_array = note_array
        self.length = len(self.flatten(note_array))
        self.dimension = self.get_list_dimension(note_array)

    # every parameter of this class should be immutable    


    # TOOLS BEGIN

    def flatten(self, note_array):
        return [
            item
            for sublist in note_array
            for item in (self.flatten(sublist.note_array) if isinstance(sublist, Cluster) else self.flatten(sublist) if isinstance(sublist, list) else [sublist])
        ]

    def chain(self, noteArray, operations) -> List[Notation]:
        return list(itertools.chain.from_iterable(operations(noteArray)))
      
    def first(self, noteArray) -> Any:
        return noteArray[0]
    
    def rest(self, noteArray) -> Any:
        return noteArray[1:]
    
    def cons(self, atom, noteArray) -> List[Any]:
        return [atom] + noteArray
    
    def emptyNote(self) -> Notation:
        return Notation(0, 1)

    def map_to_dimension_list(self, note_array, current_dimension=0):
        if isinstance(note_array, Notation):
            return current_dimension
        elif isinstance(note_array, list):
            if len(note_array) == 0:
                return current_dimension
            else:
                dimensions = []
                for item in note_array:
                    dimensions.append(self.map_to_dimension_list(item, current_dimension + 1))
                return dimensions
        elif isinstance(note_array, Cluster):
            return self.map_to_dimension_list(note_array.note_array, current_dimension)    

        else:
            raise Exception(f"Invalid type: {type(note_array)}, current_dimension={current_dimension}")
        
    def get_list_dimension(self, note_array) -> int:  
        if isinstance(note_array, list):
            if len(note_array) == 0:
                 return 1
            else:
                return max([self.get_list_dimension(elem) for elem in note_array]) + 1
        else:
            return 0

    def length_with_each_dimension(self):
        flattened_dimension_list = self.flatten(self.map_to_dimension_list(self.note_array))
        return Counter(flattened_dimension_list)

    def size_of_each_dimension(self):
        dimension_list = self.map_to_dimension_list(self.note_array)
        value_list = nested_map(lambda x: x.value, self.note_array)
        dimension_list_with_note_array = nested_zip(dimension_list, value_list)
        result = self.flatten(self.inflate(dimension_list_with_note_array))
        return Counter(result)
  

    def inflate(self, input_list):
        result = []
        for item in input_list:
            if isinstance(item, list):
                if len(item) == 2 and isinstance(item[1], int):
                    result.extend([item[0]] * item[1])
                else:
                    result.append(self.inflate(item))
            else:
                result.append(item)
        return result
    

    # TRANSFORMATIONS

    def offsetArray(self, offset) -> 'Cluster':
        return Cluster(self.note_array[offset:] + self.note_array[:offset]) 

    def scale(self, scale_factor, offset) -> 'Cluster':
        # scale_factor range: [0, 1]
        operating_array = self.offsetArray(offset)

        resolution = int(map_range(scale_factor, 0, 1, 0, self.length))
        number_to_decrease = self.length - resolution

        if is_odd(number_to_decrease):
            head = int(number_to_decrease/ 2)
            tail = number_to_decrease - head
            result = []
            for index, value in enumerate(operating_array.note_array):
                if head <= index <= operating_array.length - tail:
                    result.append(value)
            return Cluster(result)
        else :
            head = int(number_to_decrease/ 2)
            tail = head
            result = []
            for index, value in enumerate(operating_array.note_array):
                if head <= index <= self.length - tail:
                    result.append(value)
            return Cluster(result)  
            
    
    def clip(self, start, end) -> 'Cluster':
        if  start == 0 and end == 0:
            return self.copy()
        
        elif end > start and 0 <= start < self.length and 0 <= end < self.length:
            return Cluster(self.note_array[start:end])
        else:
            print("Invalid clip range" + str(start) + " " + str(end))
            return Cluster(self.note_array)
          

    def copy(self) -> 'Cluster' :
        return Cluster(self.note_array)    
    
    def loop(self, times)  -> 'Cluster':
        result = []
        for i in range(0, times):
            result.extend(self.note_array)
        return Cluster(result)
    
    def transpose(self, interval)  -> 'Cluster':
        return Cluster(list(map(lambda note: Notation(note.pitch + interval, note.value), self.note_array)))
    
    def invert(self, overlay_factor)  -> 'Cluster':
        # overlay_factor range: [0, 1]
        #left and right
        overlay_index = int(map_range(overlay_factor, 0, 1, 0, self.length))
        

        result = self.copy().clip(0, self.length - overlay_index)
        result.note_array.reverse()
        return result
    
    def mirror(self, axis_pitch, unit, method) -> 'Cluster':
        #axis range: start with middle     

        lowest_pitch = min(self.note_array, key=lambda note: note.pitch).pitch
        highest_pitch = max(self.note_array, key=lambda note: note.pitch).pitch
        middle_pitch = (lowest_pitch + highest_pitch) / 2
        mirror_axis = middle_pitch + axis_pitch * unit
        result = []
        for note in self.note_array:
            if method == "up":
                if note.pitch > mirror_axis:
                    result.append(Notation(
                        note.pitch - (note.pitch - mirror_axis) * 2,
                        note.duration
                    ))
            #up and down
            elif method == "below":
                result.append(Notation(
                    note.pitch - (note.pitch - mirror_axis) * 2,
                    note.duration
                ))
            else:
                raise Exception("Invalid method")    
        return Cluster(result)
    
    
    def re_group(self, group_size, begin_index) -> 'Cluster':
        # cluster after regroup cannot be used for further regroup
        # group_size: number of notes in each group
        # begin_index: index of the first note in the first group
        manipulate = []
        before = []
        after = []
        if begin_index + group_size < self.length:
            for index, item in enumerate(self.note_array):
                if index >= begin_index and index <= begin_index + group_size:
                    manipulate.append(item)
                elif index < begin_index:
                    before.append(item)
                elif index > begin_index + group_size:
                    after.append(item)

            result = before + [manipulate] + after                
            return Cluster(result)
        else:
            length = self.length
            print("Invalid group size or begin index, group size: {}, begin index: {}, length: {}".format(group_size, begin_index, length))
            return Cluster(self.note_array)
        
    
    
        

    def remove_duplicates_last(self) -> 'Cluster':
        result = []
        for index, item in enumerate(self.note_array):
            if index == 0 or item.pitch != self.note_array[index - 1].pitch:
                result.append(item)
            else:
                continue
                
        return Cluster(result)


    def inflat(self, inflat_num) -> 'Cluster':
        result = []
        for note in self.note_array:
            result.extend([note] * inflat_num)
        return Cluster(result)    
    
    def generate_progression(self, progression_pattern, source_safe_value: int = 10, progression_safe_value: int = 6) -> 'Cluster':
        # progression_pattern: list of intervals
        # don't over use it or it will explode
        result = []
        if self.length > source_safe_value or len(progression_pattern) > progression_safe_value:
            print("Invalid source or progression pattern, source length: {}, progression pattern length: {}".format(self.length, len(progression_pattern)))
            return Cluster(self.note_array)
        else:
            for item in self.note_array:
                for interval in progression_pattern:
                    result.append(Notation(
                        item.pitch + interval,
                        item.value
                    ))

            return Cluster(result)        
      

    def compose(self, other, command) -> 'Cluster':
        # other: another Cluster
        # command: add, sub
        result = []
        if command == "add":
            for index, item in enumerate(self.note_array):
                result.append(Notation(
                    item.pitch + other.note_array[index].pitch,
                    item.duration
                ))
            return Cluster(result)
        elif command == "sub":
            for index, item in enumerate(self.note_array):
                result.append(Notation(
                    item.pitch - other.note_array[index].pitch,
                    item.duration
                ))
            return Cluster(result)
        else:
            raise Exception("Invalid command")
        
    def focus(self, indexes) -> 'Cluster':
        result = []
        for index, item in enumerate(self.note_array):
            if index in indexes:
                print(index)
                result.append(item)
            else:
                result.append(self.emptyNote())
        return Cluster(result)        


    
    def hijack(self, index, length) -> 'Cluster':
        copy = self.copy()
        if index < self.length:
            copy.note_array[index].value = length
        else:
            print("Hijack Failed: Invalid index, index: {}, length: {}".format(index, self.length))   
        return copy
    
    def re_sequence(self, pattern) -> 'Cluster':
        copy = self.copy()
        if len(pattern) < len(self.note_array):
            for index, item in enumerate(copy.note_array):
                item.value = item.value + pattern[index]
        return copy        

    def drumify(self, interval) -> 'Cluster':
        result = []
        for item in self.note_array:
            result.append(item)
            for space in range(0, interval):
                result.append(self.emptyNote())

        return Cluster(result)
    
    def connect(self, other) -> 'Cluster':
        copy = self.copy()
        copy.note_array.extend(other.note_array)
        return copy
    
    def pad(self, interval) -> 'Cluster':
        result = []
        for space in range(0, interval):
            result.append(self.emptyNote())
        result.extend(self.note_array)
        return Cluster(result)
    
    def push(self, direction, interval) -> 'Cluster':
        if direction == "left":
            result = []
            for space in range(0, interval):
                result.append(self.emptyNote())
            result.extend(self.note_array)
            return Cluster(result)
        elif direction == "right":
            result = self.note_array
            for space in interval:
                result.append(self.emptyNote())
            return Cluster(result)    
        else:
            raise Exception("Invalid direction")
        


    # operator channel    
    
    def zip_channel(self, channel):
        channel_signature = map(lambda x: channel, self.note_array)
        return zip(self.note_array, channel_signature)
    
    def zip_sequence_channel(self, sequence, channel):
        channel_signature = map(lambda x: channel, sequence)
        return zip(sequence, channel_signature)
    
    def get_note(self, index):
        if index < self.length:
            return self.note_array[index]   
        else:
            print("get_node Failed: Invalid index, index: {}".format(index)) 
    
    
    def channel_filter(self, channel):
        result = []
        for note, channel_signature in self.zip_channel(channel):
            if note.channel == channel_signature:
                result.append(note)
        return result
    
    def split_to(self, index, method, channel_to):
        # method: before, after
        result = []
        if method == "before":
            for note in self.note_array:
                if note.index < index:
                    result.append(note)
                else:
                    break
        elif method == "after":
            for note in self.note_array:
                if note.index > index:
                    result.append(note)
        else:
            raise Exception("Invalid method")

        return self.zip_sequence_channel(result, channel_to)  
    

    #operator info
     
    def find_chord_pitches(self, root_pitch, chord_type):
        intervals = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'diminished': [0, 3, 6],
            'augmented': [0, 4, 8],
            'sus2': [0, 2, 7],
            'sus4': [0, 5, 7],
            # Add more chord types and their corresponding intervals if needed
        }

        if chord_type not in intervals:
            raise ValueError('Invalid chord type')

        chord_intervals = intervals[chord_type]
        chord_pitches = [(root_pitch + interval) % 128 for interval in chord_intervals]

        return chord_pitches
    
    def pretty_print(self):
        for note in self.note_array:
            print("pitch: {}, duration: {}".format(note.pitch, note.value))
        return self  


        


    # CALCULATE DURATION BEGIN

    def flatmap_to_note_duration_list(self):
        flattened_notes = self.flatten(self.note_array)
        flattened_dimension_list = self.flatten(self.map_to_dimension_list(self.note_array))
        dimension_length_dict = dict(self.size_of_each_dimension())

        for dimension, count in dimension_length_dict.items():
            if dimension < len(dimension_length_dict):
                dimension_length_dict[dimension] += 1

        note_durations = []
        for note, dimension in zip(flattened_notes, flattened_dimension_list):
            if dimension == 0:
                raise Exception("Invalid dimension")
            else:
                duration = note.value
                for i in range(1, dimension + 1):
                    duration /= dimension_length_dict[i]
                note_durations.append(duration)

        return note_durations
    
    # EXPORT TO NODE

      

    def get_event_bind(self) -> List[Tuple[Notation, float]]:
        notes = self.flatten(self.note_array)
        note_durations = self.flatmap_to_note_duration_list()
        return list(zip(notes, note_durations))
    
    def get_cluster_out(self) -> List[Note_Out]:
        event_list = self.get_event_bind()

        note_out_list = []

        last_end_time = 0
        for event in event_list:
            note_out_list.append(Note_Out(event[0].pitch, last_end_time, last_end_time + event[1]))
            last_end_time += event[1]
        return note_out_list    
    
    
    

class QuantizedNote:
    def __init__(self, pitch, quantized_start_step, quantized_end_step, total_quantization_step):
        self.pitch = pitch
        self.quantizedStartStep = quantized_start_step
        self.quantizedEndStep = quantized_end_step
        self.total_quantization_step = total_quantization_step

    def full_res_to_Notation(self) -> Notation:
        duration = (self.quantizedEndStep - self.quantizedStartStep) / self.total_quantization_step
        return Notation(self.pitch, duration)
    

    def to_Notation(self) -> Notation:
        return Notation(self.pitch, 1)
    

def quantized_Notes_to_Clusters(quantized_notes : List[QuantizedNote], total_quantization_step):
    noteArray = list(map(lambda note: note.to_Notation(), quantized_notes))
    return Cluster(noteArray)


def note_array_to_sequence(note_array, total_quantize_step):
    result = []
    for note in note_array:
        jsonNote = json.loads(note)
        note = QuantizedNote(pitch= jsonNote["pitch"], quantized_start_step=jsonNote["quantizedStartStep"], quantized_end_step=  jsonNote["quantizedEndStep"],total_quantization_step = 128)
        result.append(note)
    return result    
            

def convert_DAT_to_array(dat):
    '''only for 1-dimensional DAT'''
    data_array = []
    for row in range(dat.numRows):
        data_array.append(dat[row, 0].val)
    return data_array            
   
def convert_array_to_DAT(array):
    '''only for 1-dimensional DAT'''
    dat = op('table1')
    dat.clear()
    for i in range(len(array)):
        rows = dat.appendRow([array[i].pitch, array[i].start_time, array[i].end_time])
    return dat


 

jsons = convert_DAT_to_array(op('json2'))
cluster = quantized_Notes_to_Clusters(note_array_to_sequence(jsons, 128), 128)

note = cluster.remove_duplicates_last()


convert_array_to_DAT(note.get_cluster_out())